"""
Decision-state table construction for next-POI retrieval.

Offline:  build_decision_state_table()      — one labeled row per decision point
Online:   build_current_decision_state()    — one unlabeled row for the current prefix

Each row is a fixed-dimension state vector capturing:
  - Current POI spatial descriptors (copied with curr_ prefix)
  - Time-of-day
  - Prefix summary statistics
  - Recent incoming transitions (binned + raw continuous)
  - Optional Module 1 prototype signals

Raw continuous columns (curr_Latitude, curr_Longitude, prev1_gap_s,
prev1_distance_m, prev1_bearing_deg, ...) feed the vector encoder
for retrieval. Binned columns feed prompt construction.
"""

from __future__ import annotations

from typing import Optional, Union

import pickle
from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
from termcolor import cprint

from spatial_encoding.pair_transition_features_extraction import (
    compute_single_session_transitions,
    build_pair_lookup_dict,
    build_poi_coord_map,
)
from spatial_encoding.extract_poi_spatial_descriptors import SpatialEncodingConfig


def _decision_time_bin(ts: pd.Timestamp) -> str:
    hour = int(ts.hour)
    if 5 <= hour < 11:
        return "morning"
    if 11 <= hour < 15:
        return "midday"
    if 15 <= hour < 19:
        return "afternoon"
    if 19 <= hour < 24:
        return "evening"
    return "night"


# ---------------------------------------------------------------------------
# Shared helper: extract recent transition features for a decision point
# ---------------------------------------------------------------------------


def _extract_recent_transitions(
    trans_index_df,  # transitions for this session, indexed by transition_index
    current_checkin_pos,  # 0-based position of the current checkin in the session
    recent_k: int,
    gap_col: str,  # "gap_bin" or "temporal_gap_bin"
) -> dict:
    """
    Extract prev1_, prev2_, ... features for the decision point at
    current_checkin_pos. Returns both binned and raw continuous values.

    The transition *into* checkin at position p has transition_index = p - 1.
    So prev1 (most recent incoming) is at transition_index = current_checkin_pos - 1.
    """
    rec = {}

    for lag in range(1, recent_k + 1):
        prefix = f"prev{lag}"
        tr_idx = (current_checkin_pos - 1) - (lag - 1)

        if tr_idx < 0 or trans_index_df is None or tr_idx not in trans_index_df.index:
            # BOS sentinel for binned columns, NaN for raw
            rec[f"{prefix}_gap_bin"] = "BOS"
            rec[f"{prefix}_distance_bin"] = "BOS"
            rec[f"{prefix}_direction_bin"] = "BOS"
            rec[f"{prefix}_gap_s"] = np.nan
            rec[f"{prefix}_distance_m"] = np.nan
            rec[f"{prefix}_bearing_deg"] = np.nan
        else:
            tr = trans_index_df.loc[tr_idx]
            # Binned
            rec[f"{prefix}_gap_bin"] = tr[gap_col] if gap_col else None
            rec[f"{prefix}_distance_bin"] = tr["distance_bin"]
            rec[f"{prefix}_direction_bin"] = tr["direction_bin"]
            # Raw continuous
            rec[f"{prefix}_gap_s"] = float(tr.get("gap_s", np.nan))
            rec[f"{prefix}_distance_m"] = float(tr.get("final_distance_m", np.nan))
            rec[f"{prefix}_bearing_deg"] = float(tr.get("bearing_deg", np.nan))

    return rec


def _resolve_gap_col(columns) -> str:
    if "gap_bin" in columns:
        return "gap_bin"
    if "temporal_gap_bin" in columns:
        return "temporal_gap_bin"
    raise ValueError(
        "Transition data must contain either 'gap_bin' or 'temporal_gap_bin'."
    )


# ---------------------------------------------------------------------------
# Offline: build decision-state table from full training data
# ---------------------------------------------------------------------------


def build_decision_state_table(
    checkins_df: pd.DataFrame,
    poi_descriptor_df: pd.DataFrame,
    session_transitions_df: pd.DataFrame,
    config,
    *,
    session_prototype_df: Optional[pd.DataFrame] = None,
    recent_k: int = 2,
    sort_output: bool = True,
) -> pd.DataFrame:
    """
    Build one row per decision point from full session data.

    A decision point is the state after observing check-ins up to position i,
    with the actual next POI at position i+1 used as the label.
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    required_checkin_cols = [
        config.session_id_col,
        config.poi_id_col,
        config.timestamp_col,
    ]
    missing_checkin = [c for c in required_checkin_cols if c not in checkins_df.columns]
    if missing_checkin:
        raise ValueError(f"Missing required columns in checkins_df: {missing_checkin}")

    if config.poi_id_col not in poi_descriptor_df.columns:
        raise ValueError(f"poi_descriptor_df must contain {config.poi_id_col!r}")

    if config.session_id_col not in session_transitions_df.columns:
        raise ValueError(
            f"session_transitions_df must contain {config.session_id_col!r}"
        )

    if "transition_index" not in session_transitions_df.columns:
        raise ValueError("session_transitions_df must contain 'transition_index'")

    gap_col = _resolve_gap_col(session_transitions_df.columns)

    for col in ["distance_bin", "direction_bin"]:
        if col not in session_transitions_df.columns:
            raise ValueError(f"session_transitions_df must contain {col!r}")

    # ------------------------------------------------------------------
    # Normalize and sort check-ins
    # ------------------------------------------------------------------
    df = checkins_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[config.timestamp_col]):
        df[config.timestamp_col] = pd.to_datetime(
            df[config.timestamp_col], errors="coerce"
        )
    if df[config.timestamp_col].isna().any():
        bad_count = int(df[config.timestamp_col].isna().sum())
        raise ValueError(f"{bad_count} rows have invalid timestamps after parsing.")

    sort_cols = [config.session_id_col, config.timestamp_col]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Build reusable maps
    # ------------------------------------------------------------------
    poi_desc = (
        poi_descriptor_df.drop_duplicates(subset=[config.poi_id_col], keep="first")
        .set_index(config.poi_id_col)
        .copy()
    )
    poi_desc_cols = list(poi_desc.columns)

    proto_map = None
    proto_cols = []
    if session_prototype_df is not None:
        if config.session_id_col not in session_prototype_df.columns:
            raise ValueError(
                f"session_prototype_df must contain {config.session_id_col!r}"
            )
        proto_df = (
            session_prototype_df.drop_duplicates(
                subset=[config.session_id_col], keep="first"
            )
            .set_index(config.session_id_col)
            .copy()
        )
        proto_cols = list(proto_df.columns)
        proto_map = proto_df

    # Index transitions by (session_id → transition_index)
    transition_groups = {
        sid: sdf.set_index("transition_index", drop=False).copy()
        for sid, sdf in session_transitions_df.groupby(
            config.session_id_col, sort=False
        )
    }

    # ------------------------------------------------------------------
    # Build one row per decision point
    # ------------------------------------------------------------------
    records = []
    has_user = hasattr(config, "user_id_col") and config.user_id_col in df.columns
    has_category = hasattr(config, "category_col") and config.category_col in df.columns

    for session_id, sdf in df.groupby(config.session_id_col, sort=False):
        sdf = sdf.sort_values(config.timestamp_col).reset_index(drop=True)
        if len(sdf) < 2:
            continue

        trans_sdf = transition_groups.get(session_id, None)
        session_start_ts = sdf.loc[0, config.timestamp_col]

        seen_pois = set()
        seen_cats = set()

        for i in range(len(sdf) - 1):
            curr_row = sdf.iloc[i]
            next_row = sdf.iloc[i + 1]

            curr_poi = curr_row[config.poi_id_col]
            next_poi = next_row[config.poi_id_col]
            curr_ts = curr_row[config.timestamp_col]
            next_ts = next_row[config.timestamp_col]

            if curr_poi not in poi_desc.index:
                raise KeyError(
                    f"Current POI {curr_poi!r} missing from poi_descriptor_df"
                )

            seen_pois.add(curr_poi)
            if has_category and pd.notna(curr_row[config.category_col]):
                seen_cats.add(curr_row[config.category_col])

            rec = {
                config.session_id_col: session_id,
                "decision_index": i,
                "prefix_len": i + 1,
                "current_POIId": curr_poi,
                "next_POIId": next_poi,
                "current_timestamp": curr_ts,
                "next_timestamp": next_ts,
                "current_time_bin": _decision_time_bin(curr_ts),
                "prefix_elapsed_min": (
                    (curr_ts - session_start_ts).total_seconds() / 60.0
                ),
                "prefix_unique_poi_count": len(seen_pois),
                "prefix_repeat_ratio": 1.0 - (len(seen_pois) / float(i + 1)),
            }

            if has_user:
                rec[config.user_id_col] = curr_row[config.user_id_col]

            if has_category:
                rec["current_category"] = curr_row[config.category_col]
                rec["next_category"] = next_row[config.category_col]
                rec["prefix_unique_category_count"] = len(seen_cats)

            # Current POI spatial descriptors (all columns with curr_ prefix)
            curr_desc = poi_desc.loc[curr_poi]
            for col in poi_desc_cols:
                rec[f"curr_{col}"] = curr_desc[col]

            # Recent incoming transitions (binned + raw)
            rec.update(
                _extract_recent_transitions(
                    trans_index_df=trans_sdf,
                    current_checkin_pos=i,
                    recent_k=recent_k,
                    gap_col=gap_col,
                )
            )

            # Optional Module 1 prototype signals
            if proto_map is not None and session_id in proto_map.index:
                proto_row = proto_map.loc[session_id]
                if isinstance(proto_row, pd.DataFrame):
                    proto_row = proto_row.iloc[0]
                for col in proto_cols:
                    rec[f"proto_{col}"] = proto_row[col]

            records.append(rec)

    decision_state_df = pd.DataFrame(records)

    if len(decision_state_df) == 0:
        return decision_state_df

    if sort_output:
        sort_cols = [config.session_id_col, "decision_index"]
        if has_user and config.user_id_col in decision_state_df.columns:
            sort_cols = [config.user_id_col, *sort_cols]
        decision_state_df = decision_state_df.sort_values(sort_cols).reset_index(
            drop=True
        )

    return decision_state_df


# ---------------------------------------------------------------------------
# Helpers for online path
# ---------------------------------------------------------------------------


def _normalize_proto_signals(
    prototype_signals: Optional[Union[dict, pd.Series, pd.DataFrame]],
) -> dict:
    if prototype_signals is None:
        return {}
    if isinstance(prototype_signals, dict):
        return dict(prototype_signals)
    if isinstance(prototype_signals, pd.Series):
        return prototype_signals.to_dict()
    if isinstance(prototype_signals, pd.DataFrame):
        if len(prototype_signals) != 1:
            raise ValueError("prototype_signals DataFrame must have exactly one row.")
        return prototype_signals.iloc[0].to_dict()
    raise TypeError(
        "prototype_signals must be None, dict, pandas Series, or single-row DataFrame."
    )


# ---------------------------------------------------------------------------
# Online: build current decision state from a partial session
# ---------------------------------------------------------------------------


def build_current_decision_state(
    partial_session_df: pd.DataFrame,
    poi_descriptor_df: pd.DataFrame,
    config,
    *,
    pair_lookup_df: Optional[pd.DataFrame] = None,
    poi_df: Optional[pd.DataFrame] = None,
    _pair_lookup: Optional[dict] = None,
    _poi_coord_map: Optional[dict] = None,
    prototype_signals: Optional[Union[dict, pd.Series, pd.DataFrame]] = None,
    recent_k: int = 2,
) -> pd.DataFrame:
    """
    Build a single-row decision-state representation from a partial session.

    Output schema matches build_decision_state_table() minus label columns
    (next_POIId, next_timestamp, next_category).
    """
    required_cols = [
        config.session_id_col,
        config.poi_id_col,
        config.timestamp_col,
    ]
    missing = [c for c in required_cols if c not in partial_session_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in partial_session_df: {missing}")

    if config.poi_id_col not in poi_descriptor_df.columns:
        raise ValueError(f"poi_descriptor_df must contain {config.poi_id_col!r}")

    if len(partial_session_df) == 0:
        raise ValueError(
            "partial_session_df must contain at least one observed check-in."
        )

    if _pair_lookup is None:
        if pair_lookup_df is None:
            raise ValueError("Provide either _pair_lookup or pair_lookup_df.")
        _pair_lookup = build_pair_lookup_dict(pair_lookup_df)

    if _poi_coord_map is None:
        if poi_df is None:
            raise ValueError("Provide either _poi_coord_map or poi_df.")
        _poi_coord_map = build_poi_coord_map(poi_df, config)

    df = partial_session_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[config.timestamp_col]):
        df[config.timestamp_col] = pd.to_datetime(
            df[config.timestamp_col], errors="coerce"
        )
    if df[config.timestamp_col].isna().any():
        bad_count = int(df[config.timestamp_col].isna().sum())
        raise ValueError(f"{bad_count} rows have invalid timestamps after parsing.")

    df = df.sort_values(config.timestamp_col).reset_index(drop=True)

    has_user = hasattr(config, "user_id_col") and config.user_id_col in df.columns
    has_category = hasattr(config, "category_col") and config.category_col in df.columns

    # Build observed transitions on the prefix
    transition_df = compute_single_session_transitions(
        session_df=df,
        pair_lookup=_pair_lookup,
        poi_coord_map=_poi_coord_map,
        config=config,
    )

    # Current decision point = last observed row
    curr_row = df.iloc[-1]
    curr_poi = curr_row[config.poi_id_col]
    curr_ts = curr_row[config.timestamp_col]
    session_id = curr_row[config.session_id_col]

    poi_desc = (
        poi_descriptor_df.drop_duplicates(subset=[config.poi_id_col], keep="first")
        .set_index(config.poi_id_col)
        .copy()
    )

    if curr_poi not in poi_desc.index:
        raise KeyError(f"Current POI {curr_poi!r} missing from poi_descriptor_df")

    curr_desc = poi_desc.loc[curr_poi]

    rec = {
        config.session_id_col: session_id,
        "decision_index": len(df) - 1,
        "prefix_len": len(df),
        "current_POIId": curr_poi,
        "current_timestamp": curr_ts,
        "current_time_bin": _decision_time_bin(curr_ts),
        "prefix_elapsed_min": (
            (curr_ts - df.iloc[0][config.timestamp_col]).total_seconds() / 60.0
        ),
        "prefix_unique_poi_count": int(df[config.poi_id_col].nunique()),
        "prefix_repeat_ratio": 1.0 - (df[config.poi_id_col].nunique() / float(len(df))),
    }

    if has_user:
        rec[config.user_id_col] = curr_row[config.user_id_col]

    if has_category:
        rec["current_category"] = curr_row[config.category_col]
        rec["prefix_unique_category_count"] = int(
            df[config.category_col].nunique(dropna=True)
        )

    # Current POI descriptors
    for col in curr_desc.index:
        rec[f"curr_{col}"] = curr_desc[col]

    # Recent transitions (binned + raw)
    if len(transition_df) > 0:
        trans_by_idx = transition_df.set_index("transition_index", drop=False)
        gap_col = _resolve_gap_col(transition_df.columns)
    else:
        trans_by_idx = None
        gap_col = "gap_bin"

    rec.update(
        _extract_recent_transitions(
            trans_index_df=trans_by_idx,
            current_checkin_pos=len(df) - 1,
            recent_k=recent_k,
            gap_col=gap_col,
        )
    )

    # Optional prototype signals
    proto_dict = _normalize_proto_signals(prototype_signals)
    for key, val in proto_dict.items():
        rec[f"proto_{key}"] = val

    return pd.DataFrame([rec])


if __name__ == "__main__":
    config = SpatialEncodingConfig()
    city = "nyc"
    scrip_dir = Path(__file__).resolve().parent.parent

    cprint(f"\nLoading {city} train checkins dataframe...", "yellow")
    train_checkins_df = pd.read_csv(scrip_dir / f"data/{city}/train_sample.csv")

    cprint(f"\nLoading {city} poi descriptor dataframe...", "yellow")
    poi_descriptor_df = pd.read_csv(
        scrip_dir / f"artifacts/{city}/{city}_poi_descriptor.csv"
    )

    cprint(f"\nLoading {city} sessions transitions dataframe...", "yellow")
    session_transition_df = pd.read_csv(
        scrip_dir / f"artifacts/{city}/{city}_session_transition.csv"
    )

    decision_state_df = build_decision_state_table(
        train_checkins_df, poi_descriptor_df, session_transition_df, config
    )

    decision_state_df.to_csv(
        scrip_dir / f"artifacts/{city}/{city}_decision_state_table.csv",
        index=False,
    )
