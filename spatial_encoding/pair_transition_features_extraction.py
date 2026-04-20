"""
Transition descriptor computation for both batch (offline) and single-session (online) paths.

Batch:   build_all_session_transition_descriptors()  — loops over sessions
Online:  compute_single_session_transitions()        — one session

Both produce DataFrames with a shared output schema:
    session_id_col | transition_index | src_POIId | dst_POIId
    gap_bin        | distance_bin     | direction_bin

The encoder (sessions_spatial_tokens_encoder) consumes this output directly
via (session_id, transition_index) lookup — no spatial math at encoding time.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


import numpy as np
import pandas as pd
from tqdm import tqdm
from termcolor import cprint

from sparse_pair_transition_lookup import (
    _haversine_from_one_to_many_m,
    _bearing_from_one_to_many_deg,
    _bin_distances_m,
    _bearing_deg_to_direction_bin,
)
from extract_poi_spatial_descriptors import SpatialEncodingConfig


# ---------------------------------------------------------------------------
# Temporal gap binning (canonical implementation — minutes format)
# ---------------------------------------------------------------------------


def _gap_bin_minutes(gap_min: float, edges_min: tuple[float, ...]) -> str:
    """
    Bin a temporal gap (in minutes) into a human-readable label.

    Returns 'BOS' for NaN (beginning-of-session sentinel).
    """
    if pd.isna(gap_min):
        raise ValueError("Value error for gap")

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
    Convert the sparse pair-transition DataFrame into a fast dict:
        {(src_poi, dst_poi): {"distance_bin": ..., "direction_bin": ...}}
    """
    keep_cols = ["src_POIId", "dst_POIId", "distance_bin", "direction_bin"]
    available = [c for c in keep_cols if c in pair_lookup_df.columns]
    return (
        pair_lookup_df[available]
        .drop_duplicates(subset=["src_POIId", "dst_POIId"], keep="first")
        .set_index(["src_POIId", "dst_POIId"])
        .to_dict(orient="index")
    )


def build_poi_coord_map(
    poi_df: pd.DataFrame,
    config,
) -> dict:
    """
    Convert POI DataFrame into a fast dict:
        {poi_id: {lat_col: ..., lon_col: ...}}
    """
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

    Parameters
    ----------
    session_df : DataFrame
        Check-ins for a single session.
        Required columns: config.poi_id_col, config.timestamp_col.
    pair_lookup : dict
        Pre-built via build_pair_lookup_dict().
    poi_coord_map : dict
        Pre-built via build_poi_coord_map(). Used for haversine
        fallback when a pair is missing from the sparse lookup.
    config : SpatialEncodingConfig
    session_id : optional
        Explicit session ID. If None, inferred from session_df.

    Returns
    -------
    DataFrame with columns:
        config.session_id_col, transition_index,
        src_POIId, dst_POIId,
        gap_bin, distance_bin, direction_bin
    """
    output_cols = [
        config.session_id_col,
        "transition_index",
        "src_POIId",
        "dst_POIId",
        "gap_bin",
        "distance_bin",
        "direction_bin",
    ]

    df = session_df.copy()

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df[config.timestamp_col]):
        df[config.timestamp_col] = pd.to_datetime(
            df[config.timestamp_col], errors="coerce"
        )

    df = df.sort_values(config.timestamp_col).reset_index(drop=True)

    if len(df) < 2:
        return pd.DataFrame(columns=output_cols)

    # Resolve session ID
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

        # ---- Spatial: lookup first, haversine fallback ----
        pair_desc = pair_lookup.get((src_poi, dst_poi))

        if pair_desc is not None:
            distance_bin = pair_desc["distance_bin"]
            direction_bin = pair_desc["direction_bin"]
        else:
            src_c = poi_coord_map.get(src_poi)
            dst_c = poi_coord_map.get(dst_poi)

            if src_c is None or dst_c is None:
                raise KeyError(
                    f"Missing coordinates for POI {src_poi} or {dst_poi} "
                    f"in poi_coord_map. Cannot compute haversine fallback."
                )

            dist_m = _haversine_from_one_to_many_m(
                float(src_c[config.lat_col]),
                float(src_c[config.lon_col]),
                np.array([float(dst_c[config.lat_col])]),
                np.array([float(dst_c[config.lon_col])]),
            )[0]

            bearing = _bearing_from_one_to_many_deg(
                float(src_c[config.lat_col]),
                float(src_c[config.lon_col]),
                np.array([float(dst_c[config.lat_col])]),
                np.array([float(dst_c[config.lon_col])]),
            )[0]

            scaled_dist = dist_m * stretch
            distance_bin = _bin_distances_m(
                np.array([scaled_dist]), edges_m=dist_edges_m
            )[0]
            direction_bin = _bearing_deg_to_direction_bin(np.array([bearing]))[0]

        records.append(
            {
                config.session_id_col: session_id,
                "transition_index": idx - 1,
                "src_POIId": src_poi,
                "dst_POIId": dst_poi,
                "gap_bin": gap_bin,
                "distance_bin": distance_bin,
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

    This is the offline/batch entry point. It builds the lookup dicts
    once and calls compute_single_session_transitions() per session.

    Parameters
    ----------
    checkins_df : DataFrame
        All check-ins. Required columns:
        config.session_id_col, config.poi_id_col, config.timestamp_col.
    pair_lookup_df : DataFrame
        Output of build_sparse_pair_transition_lookup().
    poi_df : DataFrame
        POI table with at least poi_id, lat, lon (for haversine fallback).
    config : SpatialEncodingConfig
    show_progress : bool
        Show a tqdm progress bar over sessions.

    Returns
    -------
    DataFrame with the same schema as compute_single_session_transitions().
    """
    required_cols = [
        config.session_id_col,
        config.poi_id_col,
        config.timestamp_col,
    ]
    missing = [c for c in required_cols if c not in checkins_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in checkins_df: {missing}")

    # Build lookup dicts once
    pair_lookup = build_pair_lookup_dict(pair_lookup_df)
    poi_coord_map = build_poi_coord_map(poi_df, config)

    # Sort and parse timestamps globally
    df = checkins_df.copy()
    df[config.timestamp_col] = pd.to_datetime(df[config.timestamp_col], errors="coerce")
    df = df.sort_values([config.session_id_col, config.timestamp_col]).reset_index(drop=True)  # fmt:off

    groups = df.groupby(config.session_id_col, sort=False)
    iterator = (
        tqdm(groups, desc="Computing transitions", unit="session")
        if show_progress
        else groups
    )

    all_transitions: list[pd.DataFrame] = []

    for session_id, session_df in iterator:
        session_transitions = compute_single_session_transitions(
            session_df=session_df,
            pair_lookup=pair_lookup,
            poi_coord_map=poi_coord_map,
            config=config,
            session_id=session_id,
        )
        if not session_transitions.empty:
            all_transitions.append(session_transitions)

    if not all_transitions:
        return pd.DataFrame(
            columns=[
                config.session_id_col,
                "transition_index",
                "src_POIId",
                "dst_POIId",
                "gap_bin",
                "distance_bin",
                "direction_bin",
            ]
        )

    return pd.concat(all_transitions, ignore_index=True)


if __name__ == "__main__":
    config = SpatialEncodingConfig(
        h3_res_coarse=8,
        h3_res_fine=9,
        density_radius_m=100.0,
        timestamp_col="UTCTimeOffset",
    )

    city = "tky"
    scrip_dir = Path(__file__).resolve().parent.parent

    cprint(f"\nLoading {city} raw checkins data...", "yellow")
    checkins_df = pd.read_csv(scrip_dir / f"data/{city}/sample.csv")
    poi_df = checkins_df[["PoiId", "Latitude", "Longitude"]]
    poi_df = poi_df.drop_duplicates(subset="PoiId")
    print("Number of checkins:", len(checkins_df))
    print("Number of unique POIs:", len(poi_df))
    print("Number of unique POIs (sanity check):", checkins_df.PoiId.nunique())

    with open(scrip_dir / f"geo_data/{city}_graph.pkl", "rb") as f:
        road_graph = pickle.load(f)

    checkins_df = pd.read_csv(scrip_dir / f"data/{city}/train_sample.csv")

    pair_df = pd.read_csv(
        scrip_dir / f"artifacts/{city}/{city}_poi_pair_lookup_table.csv"
    )

    session_transition_df = build_all_session_transition_descriptors(
        checkins_df, pair_df, poi_df, config
    )

    cache_path = scrip_dir / f"artifacts/{city}/{city}_session_transition.csv"

    session_transition_df.to_csv(cache_path)
