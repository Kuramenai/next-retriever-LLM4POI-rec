from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


REQUIRED_COLUMNS = [
    "UId",
    "PId",
    "Category",
    "Latitude",
    "Longitude",
    "Time",
    "Weekday",
]


@dataclass
class CanonicalPreprocessConfig:
    dataset_root: str = "preprocessed_data"
    data: str = "NYC"
    raw_file: str | None = None
    output_dir: str | None = None

    # Step 1: raw data cleaning
    min_user_checkins: int = 10
    min_poi_unique_users: int = 5

    # Step 2: session segmentation
    session_gap_hours: float = 6.0
    min_session_len: int = 3  # warm-start setting

    # Step 3: chronological split on session start time
    train_ratio: float = 0.70
    val_ratio: float = 0.10

    timestamp_format: str = "%Y-%m-%d %H:%M:%S"

    def resolved_raw_file(self) -> Path:
        if self.raw_file is not None:
            return Path(self.raw_file)
        return Path(self.dataset_root) / self.data / f"{self.data}.csv"

    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return Path(self.output_dir)
        gap_str = f"{self.session_gap_hours:g}".replace(".", "p")
        return (
            Path(self.dataset_root)
            / self.data
            / f"canonical_gap{gap_str}h_minlen{self.min_session_len}_split701020"
        )


# -----------------------------------------------------------------------------
# I/O and validation
# -----------------------------------------------------------------------------


def validate_config(cfg: CanonicalPreprocessConfig) -> None:
    if cfg.min_user_checkins < 1:
        raise ValueError("min_user_checkins must be >= 1")
    if cfg.min_poi_unique_users < 1:
        raise ValueError("min_poi_unique_users must be >= 1")
    if cfg.session_gap_hours <= 0:
        raise ValueError("session_gap_hours must be > 0")
    if cfg.min_session_len < 1:
        raise ValueError("min_session_len must be >= 1")
    if not (0 < cfg.train_ratio < 1):
        raise ValueError("train_ratio must be in (0, 1)")
    if not (0 <= cfg.val_ratio < 1):
        raise ValueError("val_ratio must be in [0, 1)")
    if cfg.train_ratio + cfg.val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")


def read_raw_checkins(file_path: Path, timestamp_format: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"], format=timestamp_format, errors="raise")
    # Sort by user and timestamp; use PId only as a deterministic tie-breaker
    # when a user has multiple check-ins with the exact same timestamp.
    df = df.sort_values(["UId", "Time", "PId"]).reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Step 1: raw data cleaning
# -----------------------------------------------------------------------------


# def deduplicate_consecutive_identical_checkins(
#     df: pd.DataFrame,
# ) -> Tuple[pd.DataFrame, int]:
#     """
#     Drop consecutive repeated visits to the same POI for the same user.

#     Example: ... A, A, A, B, B, C -> ... A, B, C
#     Only exact consecutive duplicates are removed; non-consecutive revisits remain.
#     """
#     prev_uid = df["UId"].shift(1)
#     prev_pid = df["PId"].shift(1)
#     is_consecutive_duplicate = (df["UId"] == prev_uid) & (df["PId"] == prev_pid)

#     removed = int(is_consecutive_duplicate.sum())
#     out = df.loc[~is_consecutive_duplicate].copy().reset_index(drop=True)
#     return out, removed


def iterative_frequency_filter(
    df: pd.DataFrame,
    min_user_checkins: int,
    min_poi_unique_users: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Enforce the raw cleaning constraints until convergence.

    - Keep users with at least `min_user_checkins` total check-ins.
    - Keep POIs visited by at least `min_poi_unique_users` distinct users.

    Iteration avoids ending up with a dataset where one constraint is violated after
    applying the other. This is a principled closure of the roadmap rules, not an
    additional modeling assumption.
    """
    current = df.copy()
    stats = {
        "filter_iterations": 0,
        "removed_rows_by_user_filter": 0,
        "removed_rows_by_poi_filter": 0,
    }

    while True:
        stats["filter_iterations"] += 1
        n_before = len(current)

        user_counts = current.groupby("UId").size()
        keep_users = set(user_counts[user_counts >= min_user_checkins].index)
        after_user = current[current["UId"].isin(keep_users)].copy()
        stats["removed_rows_by_user_filter"] += n_before - len(after_user)

        poi_user_counts = after_user.groupby("PId")["UId"].nunique()
        keep_pois = set(poi_user_counts[poi_user_counts >= min_poi_unique_users].index)
        after_poi = after_user[after_user["PId"].isin(keep_pois)].copy()
        stats["removed_rows_by_poi_filter"] += len(after_user) - len(after_poi)

        current = after_poi.sort_values(["UId", "Time", "PId"]).reset_index(drop=True)
        if len(current) == n_before:
            break

    return current, stats


# -----------------------------------------------------------------------------
# Step 2: session segmentation
# -----------------------------------------------------------------------------


def assign_sessions(df: pd.DataFrame, session_gap_hours: float) -> pd.DataFrame:
    out = df.sort_values(["UId", "Time", "PId"]).copy().reset_index(drop=True)

    time_delta = out.groupby("UId")["Time"].diff()
    gap_threshold = pd.Timedelta(hours=session_gap_hours)
    is_new_session = time_delta.isna() | (time_delta > gap_threshold)

    out["user_session_index"] = (
        is_new_session.groupby(out["UId"]).cumsum().astype("int64") - 1
    )
    session_key = out["UId"].astype(str) + "__" + out["user_session_index"].astype(str)
    out["SessionId"] = pd.factorize(session_key, sort=False)[0].astype("int64")
    out["PrevGapHours"] = (time_delta.dt.total_seconds() / 3600.0).fillna(0.0)
    return out


def build_session_boundaries(df: pd.DataFrame) -> pd.DataFrame:
    session_df = (
        df.groupby("SessionId", as_index=False)
        .agg(
            UId=("UId", "first"),
            user_session_index=("user_session_index", "first"),
            session_start_time=("Time", "min"),
            session_end_time=("Time", "max"),
            session_len=("Time", "size"),
            first_poi=("PId", "first"),
            last_poi=("PId", "last"),
        )
        .sort_values(["session_start_time", "SessionId"])
        .reset_index(drop=True)
    )

    session_df["session_duration_hours"] = (
        session_df["session_end_time"] - session_df["session_start_time"]
    ).dt.total_seconds() / 3600.0
    return session_df


def filter_short_sessions(
    checkins_df: pd.DataFrame,
    sessions_df: pd.DataFrame,
    min_session_len: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    keep_session_ids = set(
        sessions_df.loc[sessions_df["session_len"] >= min_session_len, "SessionId"]
    )
    filtered_checkins = (
        checkins_df[checkins_df["SessionId"].isin(keep_session_ids)]
        .copy()
        .reset_index(drop=True)
    )
    filtered_sessions = (
        sessions_df[sessions_df["SessionId"].isin(keep_session_ids)]
        .copy()
        .reset_index(drop=True)
    )
    removed_sessions = int(len(sessions_df) - len(filtered_sessions))
    return filtered_checkins, filtered_sessions, removed_sessions


# -----------------------------------------------------------------------------
# Step 3: chronological split on session start time
# -----------------------------------------------------------------------------


def split_sessions_chronologically(
    sessions_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> pd.DataFrame:
    out = (
        sessions_df.sort_values(["session_start_time", "SessionId"])
        .copy()
        .reset_index(drop=True)
    )
    n = len(out)

    train_end = math.floor(n * train_ratio)
    val_end = math.floor(n * (train_ratio + val_ratio))

    out["SplitTag"] = "test"
    if train_end > 0:
        out.loc[: train_end - 1, "SplitTag"] = "train"
    if val_end > train_end:
        out.loc[train_end : val_end - 1, "SplitTag"] = "validation"
    return out


def attach_split_tags(
    checkins_df: pd.DataFrame, sessions_df: pd.DataFrame
) -> pd.DataFrame:
    session_split = sessions_df[["SessionId", "SplitTag"]].copy()
    out = checkins_df.merge(
        session_split, on="SessionId", how="left", validate="many_to_one"
    )
    if out["SplitTag"].isna().any():
        raise ValueError("Found check-ins without a session split assignment.")

    out = out.sort_values(["UId", "Time", "PId"]).reset_index(drop=True)
    out["session_checkin_index"] = out.groupby("SessionId").cumcount().astype("int64")
    out["user_checkin_index"] = out.groupby("UId").cumcount().astype("int64")
    return out


# -----------------------------------------------------------------------------
# Artifact writing
# -----------------------------------------------------------------------------


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_artifacts(
    cfg: CanonicalPreprocessConfig,
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    sessions_df: pd.DataFrame,
    final_df: pd.DataFrame,
    stats: Dict[str, int],
) -> Dict[str, str]:
    out_dir = cfg.resolved_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_sessions = sessions_df[sessions_df["SplitTag"] == "train"].copy()
    val_sessions = sessions_df[sessions_df["SplitTag"] == "validation"].copy()
    test_sessions = sessions_df[sessions_df["SplitTag"] == "test"].copy()

    train_checkins = final_df[final_df["SplitTag"] == "train"].copy()
    val_checkins = final_df[final_df["SplitTag"] == "validation"].copy()
    test_checkins = final_df[final_df["SplitTag"] == "test"].copy()

    paths = {
        "step1_clean_checkins": out_dir / "step1_clean_checkins.csv",
        "step2_all_sessions_checkins": out_dir / "step2_all_sessions_checkins.csv",
        "step2_session_boundaries": out_dir / "step2_session_boundaries.csv",
        "step3_all_checkins": out_dir / "step3_all_checkins.csv",
        "step3_train_checkins": out_dir / "step3_train_checkins.csv",
        "step3_validation_checkins": out_dir / "step3_validation_checkins.csv",
        "step3_test_checkins": out_dir / "step3_test_checkins.csv",
        "step3_all_sessions": out_dir / "step3_all_sessions.csv",
        "step3_train_sessions": out_dir / "step3_train_sessions.csv",
        "step3_validation_sessions": out_dir / "step3_validation_sessions.csv",
        "step3_test_sessions": out_dir / "step3_test_sessions.csv",
        "metadata": out_dir / "metadata.json",
    }

    # Step 1 artifact
    write_csv(clean_df, paths["step1_clean_checkins"])

    # Step 2 artifacts (warm-start sessions retained)
    write_csv(
        final_df.drop(columns=["SplitTag"]),
        paths["step2_all_sessions_checkins"],
    )
    write_csv(
        sessions_df.drop(columns=["SplitTag"]),
        paths["step2_session_boundaries"],
    )

    # Step 3 artifacts
    write_csv(final_df, paths["step3_all_checkins"])
    write_csv(train_checkins, paths["step3_train_checkins"])
    write_csv(val_checkins, paths["step3_validation_checkins"])
    write_csv(test_checkins, paths["step3_test_checkins"])
    write_csv(sessions_df, paths["step3_all_sessions"])
    write_csv(train_sessions, paths["step3_train_sessions"])
    write_csv(val_sessions, paths["step3_validation_sessions"])
    write_csv(test_sessions, paths["step3_test_sessions"])

    metadata = {
        "config": asdict(cfg),
        "raw_input_file": str(cfg.resolved_raw_file()),
        "output_dir": str(out_dir),
        "n_raw_checkins": int(len(raw_df)),
        "n_clean_checkins": int(len(clean_df)),
        "n_final_checkins": int(len(final_df)),
        "n_final_sessions": int(len(sessions_df)),
        "n_users_final": int(final_df["UId"].nunique()) if len(final_df) else 0,
        "n_pois_final": int(final_df["PId"].nunique()) if len(final_df) else 0,
        "n_train_sessions": int(len(train_sessions)),
        "n_validation_sessions": int(len(val_sessions)),
        "n_test_sessions": int(len(test_sessions)),
        "n_train_checkins": int(len(train_checkins)),
        "n_validation_checkins": int(len(val_checkins)),
        "n_test_checkins": int(len(test_checkins)),
        **stats,
    }

    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return {k: str(v) for k, v in paths.items()}


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------


def run_pipeline(cfg: CanonicalPreprocessConfig) -> Dict[str, str]:
    validate_config(cfg)
    raw_path = cfg.resolved_raw_file()
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    raw_df = read_raw_checkins(raw_path, cfg.timestamp_format)

    # dedup_df, n_removed_consecutive_duplicates = (
    #     deduplicate_consecutive_identical_checkins(raw_df)
    # )
    clean_df, filter_stats = iterative_frequency_filter(
        raw_df,
        min_user_checkins=cfg.min_user_checkins,
        min_poi_unique_users=cfg.min_poi_unique_users,
    )

    sessionized_df = assign_sessions(clean_df, session_gap_hours=cfg.session_gap_hours)
    sessions_df = build_session_boundaries(sessionized_df)
    warm_checkins_df, warm_sessions_df, n_removed_short_sessions = (
        filter_short_sessions(
            sessionized_df,
            sessions_df,
            min_session_len=cfg.min_session_len,
        )
    )

    split_sessions_df = split_sessions_chronologically(
        warm_sessions_df,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
    )
    final_df = attach_split_tags(warm_checkins_df, split_sessions_df)

    stats = {
        # "removed_consecutive_duplicates": int(n_removed_consecutive_duplicates),
        "removed_short_sessions": int(n_removed_short_sessions),
        **filter_stats,
    }

    return save_artifacts(
        cfg=cfg,
        raw_df=raw_df,
        clean_df=clean_df,
        sessions_df=split_sessions_df,
        final_df=final_df,
        stats=stats,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Canonical preprocessing for the roadmap's Steps 1-3: raw data cleaning, "
            "session segmentation, and session-level chronological split."
        )
    )
    parser.add_argument("--dataset-root", type=str, default="preprocessed_data")
    parser.add_argument("--data", type=str, default="NYC")
    parser.add_argument("--raw-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--min-user-checkins", type=int, default=10)
    parser.add_argument("--min-poi-unique-users", type=int, default=5)
    parser.add_argument("--session-gap-hours", type=float, default=6.0)
    parser.add_argument("--min-session-len", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--timestamp-format", type=str, default="%Y-%m-%d %H:%M:%S")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = CanonicalPreprocessConfig(
        dataset_root=args.dataset_root,
        data=args.data,
        raw_file=args.raw_file,
        output_dir=args.output_dir,
        min_user_checkins=args.min_user_checkins,
        min_poi_unique_users=args.min_poi_unique_users,
        session_gap_hours=args.session_gap_hours,
        min_session_len=args.min_session_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        timestamp_format=args.timestamp_format,
    )

    artifact_paths = run_pipeline(cfg)
    print("Canonical preprocessing completed.\nArtifacts:")
    for name, path in artifact_paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
