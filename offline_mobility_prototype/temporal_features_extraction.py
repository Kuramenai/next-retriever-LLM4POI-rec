import numpy as np
import pandas as pd
from typing import Dict, Tuple
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from canonical_preprocess import write_csv

TIME_BINS = ("morning", "afternoon", "evening", "late_night")


def get_time_of_day_bin(ts: pd.Timestamp) -> Tuple[int, str]:
    """
    morning:    05:00-11:59
    afternoon:  12:00-16:59
    evening:    17:00-21:59
    late_night: 22:00-04:59
    """
    hour = ts.hour

    if 5 <= hour < 12:
        return 0, "morning"
    if 12 <= hour < 17:
        return 1, "afternoon"
    if 17 <= hour < 22:
        return 2, "evening"
    return 3, "late_night"


def extract_temporal_features(
    session_df: pd.DataFrame,
    duration_mean: float | None = None,
    duration_std: float | None = None,
) -> Dict[str, object]:
    """
    Build the temporal block for a single session using all check-ins in that session.

    Output vector:
      [tod_morning_prop,
       tod_afternoon_prop,
       tod_evening_prop,
       tod_late_night_prop,
       start_is_weekend,
       duration_feature]

    Notes:
    - time-of-day is encoded as check-in-count proportions over the 4 bins
    - weekday/weekend is still taken from session start time
    - duration = session_end_time - session_start_time
    - duration is log1p-transformed, then optionally z-scored
    """
    if session_df.empty:
        raise ValueError("session_df cannot be empty")

    s = session_df.copy()
    s["Time"] = pd.to_datetime(s["Time"])
    s = s.sort_values(["Time", "PId"]).reset_index(drop=True)

    start_time = s.iloc[0]["Time"]
    end_time = s.iloc[-1]["Time"]

    # ----- 1) Time-of-day proportions from all check-ins -----
    bin_indices = (
        s["Time"].apply(lambda ts: get_time_of_day_bin(pd.Timestamp(ts))[0]).to_numpy()
    )
    tod_counts = np.bincount(bin_indices, minlength=4).astype(np.float32)
    tod_props = tod_counts / max(tod_counts.sum(), 1.0)

    # ----- 2) Weekday/weekend from session start -----
    start_is_weekend = float(start_time.weekday() >= 5)

    # ----- 3) Duration scalar -----
    duration_minutes = max((end_time - start_time).total_seconds() / 60.0, 0.0)
    duration_log = np.log1p(duration_minutes)

    if duration_mean is not None and duration_std is not None:
        safe_std = duration_std if duration_std > 1e-8 else 1.0
        duration_feature = (duration_log - duration_mean) / safe_std
    else:
        duration_feature = duration_log

    vector = np.concatenate(
        [tod_props, np.array([start_is_weekend, duration_feature], dtype=np.float32)]
    ).astype(np.float32)

    return {
        "SessionId": s.iloc[0]["SessionId"] if "SessionId" in s.columns else None,
        "session_start_time": start_time,
        "session_end_time": end_time,
        "tod_counts": tod_counts,
        "tod_props": tod_props,
        "start_is_weekend": start_is_weekend,
        "duration_minutes": duration_minutes,
        "duration_log": duration_log,
        "duration_feature": float(duration_feature),
        "vector": vector,
    }


def fit_duration_normalizer_from_checkins(
    session_checkins_df: pd.DataFrame,
) -> Tuple[float, float]:
    """
    Fit duration normalization on TRAIN sessions only.
    Expects columns: SessionId, Time
    """
    df = session_checkins_df.copy()
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values(["SessionId", "Time", "PId"]).reset_index(drop=True)

    bounds = df.groupby("SessionId", as_index=False).agg(
        session_start_time=("Time", "min"),
        session_end_time=("Time", "max"),
    )

    duration_minutes = (
        (bounds["session_end_time"] - bounds["session_start_time"]).dt.total_seconds()
        / 60.0
    ).clip(lower=0.0)

    duration_logs = np.log1p(duration_minutes.to_numpy(dtype=np.float32))
    return float(duration_logs.mean()), float(duration_logs.std())


def build_temporal_feature_matrix(
    session_checkins_df: pd.DataFrame,
    duration_mean: float | None = None,
    duration_std: float | None = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Batch extraction from session-level check-in traces.
    Expects columns: SessionId, Time, PId
    """
    
    df = session_checkins_df.copy()
    df = df.rename(columns={"pseudo_session_trajectory_id": "SessionId", "PoiCategoryId": "PId"})
    df["Time"] = pd.to_datetime(df["UTCTimeOffset"])
    df = df.sort_values(["SessionId", "Time", "PId"]).reset_index(drop=True)

    vectors = []
    rows = []

    for session_id, group in df.groupby("SessionId", sort=False):
        feat = extract_temporal_features(
            group,
            duration_mean=duration_mean,
            duration_std=duration_std,
        )

        vectors.append(feat["vector"])
        rows.append(
            {
                "SessionId": session_id,
                "session_start_time": feat["session_start_time"],
                "session_end_time": feat["session_end_time"],
                "tod_morning_prop": float(feat["tod_props"][0]),
                "tod_afternoon_prop": float(feat["tod_props"][1]),
                "tod_evening_prop": float(feat["tod_props"][2]),
                "tod_late_night_prop": float(feat["tod_props"][3]),
                "start_is_weekend": feat["start_is_weekend"],
                "duration_minutes": feat["duration_minutes"],
                "duration_log": feat["duration_log"],
                "duration_feature": feat["duration_feature"],
            }
        )

    X = np.vstack(vectors).astype(np.float32)
    meta = pd.DataFrame(rows)
    return X, meta