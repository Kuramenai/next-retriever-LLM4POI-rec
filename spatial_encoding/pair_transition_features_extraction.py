"""
Transition descriptor computation for both batch (offline) and single-session (online) paths.

Batch:   build_all_session_transition_descriptors()  — loops over sessions
Online:  compute_single_session_transitions()        — one session

Both produce DataFrames with a shared output schema containing:
    Binned columns  (for prompt construction):  gap_bin, distance_bin, direction_bin
    Raw columns     (for vector encoder):       gap_s, final_distance_m, bearing_deg
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from spatial_encoding.sparse_pair_transition_lookup import (
    _haversine_from_one_to_many_m,
    _bearing_from_one_to_many_deg,
    _bin_distances_m,
    _bearing_deg_to_direction_bin,
)


# ---------------------------------------------------------------------------
# Temporal gap binning (canonical implementation — minutes format)
# ---------------------------------------------------------------------------


def _gap_bin_minutes(gap_min: float, edges_min: tuple[float, ...]) -> str:
    if pd.isna(gap_min):
        return "BOS"
    lower = 0
    for edge in edges_min:
        if gap_min <= edge:
            return f"{int(lower)}-{int(edge)}min"
        lower = edge
    return f"{int(edges_min[-1])}+min"


# ---------------------------------------------------------------------------
# Lookup-dict builders (called once, reused across sessions)
# ---------------------------------------------------------------------------


def build_pair_lookup_dict(
    pair_lookup_df: pd.DataFrame,
) -> dict[tuple, dict]:
    """
    Convert sparse pair-transition DataFrame into a fast dict including
    both binned and raw continuous values.
    """
    keep = ["src_POIId", "dst_POIId"]
    for col in [
        "distance_bin",
        "direction_bin",
        "final_distance_m",
        "bearing_deg",
        "haversine_distance_m",
    ]:
        if col in pair_lookup_df.columns:
            keep.append(col)

    return (
        pair_lookup_df[keep]
        .drop_duplicates(subset=["src_POIId", "dst_POIId"], keep="first")
        .set_index(["src_POIId", "dst_POIId"])
        .to_dict(orient="index")
    )


def build_poi_coord_map(
    poi_df: pd.DataFrame,
    config,
) -> dict:
    return (
        poi_df[[config.poi_id_col, config.lat_col, config.lon_col]]
        .drop_duplicates(subset=[config.poi_id_col], keep="first")
        .set_index(config.poi_id_col)
        .to_dict(orient="index")
    )


# ---------------------------------------------------------------------------
# Core: single-session transition computation
# ---------------------------------------------------------------------------


def compute_single_session_transitions(
    session_df: pd.DataFrame,
    pair_lookup: dict[tuple, dict],
    poi_coord_map: dict,
    config,
    *,
    session_id: Any = None,
) -> pd.DataFrame:
    """
    Compute transition descriptors for exactly one session.

    Returns both binned and raw continuous columns per transition.
    """
    output_cols = [
        config.session_id_col,
        "transition_index",
        "src_POIId",
        "dst_POIId",
        "gap_s",
        "gap_bin",
        "final_distance_m",
        "distance_bin",
        "bearing_deg",
        "direction_bin",
    ]

    df = session_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[config.timestamp_col]):
        df[config.timestamp_col] = pd.to_datetime(
            df[config.timestamp_col], errors="coerce"
        )
    df = df.sort_values(config.timestamp_col).reset_index(drop=True)

    if len(df) < 2:
        return pd.DataFrame(columns=output_cols)

    if session_id is None:
        if config.session_id_col in df.columns:
            session_id = df[config.session_id_col].iloc[0]
        else:
            session_id = "__UNKNOWN__"

    gap_edges_min = tuple(config.gap_bin_edges_min)
    dist_edges_m = tuple(config.distance_bin_edges_m)
    stretch = float(config.road_distance_stretch_factor)

    records: list[dict] = []

    for idx in range(1, len(df)):
        prev = df.iloc[idx - 1]
        curr = df.iloc[idx]

        src_poi = prev[config.poi_id_col]
        dst_poi = curr[config.poi_id_col]

        # ---- Temporal gap ----
        gap_s = (
            curr[config.timestamp_col] - prev[config.timestamp_col]
        ).total_seconds()
        gap_bin = _gap_bin_minutes(gap_s / 60.0, gap_edges_min)

        # ---- Spatial: try pair lookup, then haversine fallback ----
        pair_desc = pair_lookup.get((src_poi, dst_poi))

        if pair_desc is not None:
            distance_bin = pair_desc.get("distance_bin")
            direction_bin = pair_desc.get("direction_bin")
            final_distance_m = pair_desc.get("final_distance_m", np.nan)
            bearing_deg = pair_desc.get("bearing_deg", np.nan)
        else:
            distance_bin = None
            direction_bin = None
            final_distance_m = np.nan
            bearing_deg = np.nan

        # Compute from coordinates if raw values missing
        needs_compute = (
            np.isnan(final_distance_m) or np.isnan(bearing_deg) or distance_bin is None
        )

        if needs_compute:
            src_c = poi_coord_map.get(src_poi)
            dst_c = poi_coord_map.get(dst_poi)
            if src_c is None or dst_c is None:
                raise KeyError(f"Missing coordinates for POI {src_poi} or {dst_poi}.")

            hav_m = _haversine_from_one_to_many_m(
                float(src_c[config.lat_col]),
                float(src_c[config.lon_col]),
                np.array([float(dst_c[config.lat_col])]),
                np.array([float(dst_c[config.lon_col])]),
            )[0]

            comp_bearing = _bearing_from_one_to_many_deg(
                float(src_c[config.lat_col]),
                float(src_c[config.lon_col]),
                np.array([float(dst_c[config.lat_col])]),
                np.array([float(dst_c[config.lon_col])]),
            )[0]

            if np.isnan(final_distance_m):
                final_distance_m = hav_m * stretch
            if np.isnan(bearing_deg):
                bearing_deg = comp_bearing

            distance_bin = _bin_distances_m(
                np.array([final_distance_m]), edges_m=dist_edges_m
            )[0]
            direction_bin = _bearing_deg_to_direction_bin(np.array([bearing_deg]))[0]

        records.append(
            {
                config.session_id_col: session_id,
                "transition_index": idx - 1,
                "src_POIId": src_poi,
                "dst_POIId": dst_poi,
                "gap_s": gap_s,
                "gap_bin": gap_bin,
                "final_distance_m": final_distance_m,
                "distance_bin": distance_bin,
                "bearing_deg": bearing_deg,
                "direction_bin": direction_bin,
            }
        )

    return pd.DataFrame(records, columns=output_cols)


# ---------------------------------------------------------------------------
# Batch: all sessions
# ---------------------------------------------------------------------------


def build_all_session_transition_descriptors(
    checkins_df: pd.DataFrame,
    pair_lookup_df: pd.DataFrame,
    poi_df: pd.DataFrame,
    config,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Compute transition descriptors for every session in checkins_df.
    Builds lookup dicts once and calls compute_single_session_transitions()
    per session.
    """
    required_cols = [
        config.session_id_col,
        config.poi_id_col,
        config.timestamp_col,
    ]
    missing = [c for c in required_cols if c not in checkins_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in checkins_df: {missing}")

    pair_lookup = build_pair_lookup_dict(pair_lookup_df)
    poi_coord_map = build_poi_coord_map(poi_df, config)

    df = checkins_df.copy()
    df[config.timestamp_col] = pd.to_datetime(df[config.timestamp_col], errors="coerce")
    df = df.sort_values([config.session_id_col, config.timestamp_col]).reset_index(
        drop=True
    )

    groups = df.groupby(config.session_id_col, sort=False)
    iterator = (
        tqdm(groups, desc="Computing transitions", unit="session")
        if show_progress
        else groups
    )

    all_transitions: list[pd.DataFrame] = []
    for session_id, session_df in iterator:
        t = compute_single_session_transitions(
            session_df=session_df,
            pair_lookup=pair_lookup,
            poi_coord_map=poi_coord_map,
            config=config,
            session_id=session_id,
        )
        if not t.empty:
            all_transitions.append(t)

    if not all_transitions:
        return pd.DataFrame(
            columns=[
                config.session_id_col,
                "transition_index",
                "src_POIId",
                "dst_POIId",
                "gap_s",
                "gap_bin",
                "final_distance_m",
                "distance_bin",
                "bearing_deg",
                "direction_bin",
            ]
        )

    return pd.concat(all_transitions, ignore_index=True)
