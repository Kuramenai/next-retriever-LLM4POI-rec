from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


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
    Build one row per decision point / observed prefix.

    A decision point is the state after observing check-ins up to position i,
    with the actual next POI at position i+1 used as the label.

    Parameters
    ----------
    checkins_df:
        Full ordered session check-ins.
        Required:
          - config.session_id_col
          - config.poi_id_col
          - config.timestamp_col
        Optional:
          - config.user_id_col
          - config.category_col

    poi_descriptor_df:
        Output of build_poi_spatial_descriptors(...).
        Must contain config.poi_id_col.
        Any available descriptor columns will be copied with prefix "curr_".

    session_transitions_df:
        Output of build_all_session_transition_descriptors(...)
        or equivalent observed-transition table.

        Expected columns:
          - session_id
          - transition_index
          - gap_bin or temporal_gap_bin
          - distance_bin
          - direction_bin

    session_prototype_df:
        Optional session-level Module 1 output.
        Must contain config.session_id_col.
        All remaining columns are copied onto every decision row
        with prefix "proto_".

    recent_k:
        Number of most recent incoming transitions to expose as fixed columns.
        Example with recent_k=2:
          - prev1_gap_bin, prev1_distance_bin, prev1_direction_bin
          - prev2_gap_bin, prev2_distance_bin, prev2_direction_bin

    Returns
    -------
    decision_state_df:
        One row per decision point with:
          - current state features
          - short recent transition summary
          - next POI label
    """
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
        raise ValueError(f"session_transitions_df must contain {config.session_id_col!r}")  # fmt: skip

    if "transition_index" not in session_transitions_df.columns:
        raise ValueError("session_transitions_df must contain 'transition_index'")

    gap_col = None
    if "gap_bin" in session_transitions_df.columns:
        gap_col = "gap_bin"
    elif "temporal_gap_bin" in session_transitions_df.columns:
        gap_col = "temporal_gap_bin"
    else:
        raise ValueError(
            "session_transitions_df must contain either 'gap_bin' or 'temporal_gap_bin'"
        )

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
    if hasattr(config, "user_id_col") and config.user_id_col in df.columns:
        sort_cols = [config.user_id_col, *sort_cols]

    df = df.sort_values(sort_cols).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Build reusable maps
    # ------------------------------------------------------------------
    poi_desc = (
        poi_descriptor_df.drop_duplicates(subset=[config.poi_id_col], keep="first")
        .set_index(config.poi_id_col)
        .copy()
    )

    poi_desc_cols = [c for c in poi_desc.columns]

    proto_map = None
    proto_cols = []
    if session_prototype_df is not None:
        if config.session_id_col not in session_prototype_df.columns:
            raise ValueError(f"session_prototype_df must contain {config.session_id_col!r}")  # fmt: skip
        proto_df = (
            session_prototype_df.drop_duplicates(
                subset=[config.session_id_col], keep="first"
            )
            .set_index(config.session_id_col)
            .copy()
        )
        proto_cols = [c for c in proto_df.columns]
        proto_map = proto_df

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

        # Need at least one future POI to define a decision point
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
                "prefix_elapsed_min": (curr_ts - session_start_ts).total_seconds()
                / 60.0,
                "prefix_unique_poi_count": len(seen_pois),
                "prefix_repeat_ratio": 1.0 - (len(seen_pois) / float(i + 1)),
            }

            if has_user:
                rec[config.user_id_col] = curr_row[config.user_id_col]

            if has_category:
                rec["current_category"] = curr_row[config.category_col]
                rec["next_category"] = next_row[config.category_col]
                rec["prefix_unique_category_count"] = len(seen_cats)

            # Current POI spatial descriptor block
            curr_desc = poi_desc.loc[curr_poi]
            for col in poi_desc_cols:
                rec[f"curr_{col}"] = curr_desc[col]

            # Short recent incoming-transition summary
            for lag in range(1, recent_k + 1):
                prefix = f"prev{lag}"

                if i - lag < 0 or trans_sdf is None or (i - lag) not in trans_sdf.index:
                    rec[f"{prefix}_gap_bin"] = "BOS"
                    rec[f"{prefix}_distance_bin"] = "BOS"
                    rec[f"{prefix}_direction_bin"] = "BOS"
                else:
                    tr = trans_sdf.loc[i - lag]
                    rec[f"{prefix}_gap_bin"] = tr[gap_col]
                    rec[f"{prefix}_distance_bin"] = tr["distance_bin"]
                    rec[f"{prefix}_direction_bin"] = tr["direction_bin"]

            # Optional Module 1 prototype block
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
