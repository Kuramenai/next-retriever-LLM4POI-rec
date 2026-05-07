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

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from termcolor import cprint

from sparse_pair_transition_lookup import (
    _bin_distances_m,
    _bearing_deg_to_direction_bin,
)

from extract_poi_spatial_descriptors import SpatialEncodingConfig


# ---------------------------------------------------------------------------
# Vectorized pairwise geo helpers (avoid Python loops for per-transition compute)
# ---------------------------------------------------------------------------


def _haversine_pairwise_m(
    lat1_deg: np.ndarray,
    lon1_deg: np.ndarray,
    lat2_deg: np.ndarray,
    lon2_deg: np.ndarray,
) -> np.ndarray:
    """
    Pairwise haversine distance for arrays of equal length.
    Returns meters.
    """
    lat1 = np.deg2rad(lat1_deg.astype(float))
    lon1 = np.deg2rad(lon1_deg.astype(float))
    lat2 = np.deg2rad(lat2_deg.astype(float))
    lon2 = np.deg2rad(lon2_deg.astype(float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return 6371000.0 * c


def _bearing_pairwise_deg(
    lat1_deg: np.ndarray,
    lon1_deg: np.ndarray,
    lat2_deg: np.ndarray,
    lon2_deg: np.ndarray,
) -> np.ndarray:
    """
    Pairwise initial bearing for arrays of equal length.
    Returns degrees in [0, 360).
    """
    lat1 = np.deg2rad(lat1_deg.astype(float))
    lon1 = np.deg2rad(lon1_deg.astype(float))
    lat2 = np.deg2rad(lat2_deg.astype(float))
    lon2 = np.deg2rad(lon2_deg.astype(float))

    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.rad2deg(np.arctan2(y, x))
    return (brng + 360.0) % 360.0


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
    keep = [
        "src_POIId",
        "dst_POIId",
        "distance_bin",
        "direction_bin",
        "final_distance_m",
        "bearing_deg",
        "haversine_distance_m",
    ]

    missing = [c for c in keep if c not in pair_lookup_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in pair_lookup_df: {missing}")
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
    lookup_df: pd.DataFrame,
    coord_df: pd.DataFrame,
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
        df[config.timestamp_col] = pd.to_datetime(df[config.timestamp_col], errors="coerce")  # fmt: skip
    df = df.sort_values(config.timestamp_col).reset_index(drop=True)

    if len(df) < 2:
        cprint("[DEBUG] len(df) < 2, returning empty DataFrame...", "red")
        return pd.DataFrame(columns=output_cols)

    if session_id is None:
        if config.session_id_col in df.columns:
            session_id = df[config.session_id_col].iloc[0]
        else:
            cprint("[DEBUG] session_id is None, setting to __UNKNOWN__...", "red")
            session_id = "__UNKNOWN__"

    gap_edges_min = tuple(config.gap_bin_edges_min)
    dist_edges_m = tuple(config.distance_bin_edges_m)
    stretch = float(config.road_distance_stretch_factor)

    # Build transitions (length n-1) without per-row iteration
    ts = df[config.timestamp_col].to_numpy()
    src_poi = df[config.poi_id_col].to_numpy()[:-1]
    dst_poi = df[config.poi_id_col].to_numpy()[1:]
    transition_index = np.arange(len(df) - 1, dtype=int)

    gap_s = (ts[1:] - ts[:-1]) / np.timedelta64(1, "s")
    gap_min = gap_s / 60.0
    gap_bin = pd.Series(gap_min).apply(lambda x: _gap_bin_minutes(x, gap_edges_min)).to_numpy()  # fmt: skip

    # ---- Lookup alignment (vectorized) ----
    pair_idx = pd.MultiIndex.from_arrays([src_poi, dst_poi], names=["src_POIId", "dst_POIId"])  # fmt: skip
    # if len(pair_lookup) > 0:
    #     lookup_df = pd.DataFrame.from_dict(pair_lookup, orient="index")
    #     # keys are (src, dst) tuples → convert to MultiIndex for fast reindex
    #     lookup_df.index = pd.MultiIndex.from_tuples(lookup_df.index, names=["src_POIId", "dst_POIId"])  # fmt: skip
    #     aligned = lookup_df.reindex(pair_idx)
    # else:
    #     aligned = pd.DataFrame(index=pair_idx)
    aligned = lookup_df.reindex(pair_idx)

    final_distance_m = aligned.get("final_distance_m", pd.Series(index=pair_idx, dtype=float)).to_numpy()  # fmt: skip
    bearing_deg = aligned.get("bearing_deg", pd.Series(index=pair_idx, dtype=float)).to_numpy()  # fmt: skip
    distance_bin = aligned.get("distance_bin", pd.Series(index=pair_idx, dtype=object)).to_numpy()  # fmt: skip
    direction_bin = aligned.get("direction_bin", pd.Series(index=pair_idx, dtype=object)).to_numpy()  # fmt: skip

    # ---- Recompute missing spatial values in bulk (haversine + bearing) ----
    needs_distance = pd.isna(final_distance_m)
    needs_bearing = pd.isna(bearing_deg)
    needs_dist_bin = pd.isna(distance_bin)
    needs_dir_bin = pd.isna(direction_bin)
    needs_any = needs_distance | needs_bearing | needs_dist_bin | needs_dir_bin

    if np.any(needs_any):
        src_coords = coord_df.reindex(src_poi)
        dst_coords = coord_df.reindex(dst_poi)
        if src_coords[[config.lat_col, config.lon_col]].isna().any(axis=None) or dst_coords[
            [config.lat_col, config.lon_col]
        ].isna().any(axis=None):
            missing_src = src_coords[src_coords[[config.lat_col, config.lon_col]].isna().any(axis=1)].index.unique().tolist()  # fmt: skip
            missing_dst = dst_coords[dst_coords[[config.lat_col, config.lon_col]].isna().any(axis=1)].index.unique().tolist()  # fmt: skip
            raise KeyError(
                "Missing coordinates for one or more POIs. "
                f"missing_src={missing_src[:10]} missing_dst={missing_dst[:10]}"
            )

        src_lat = src_coords[config.lat_col].to_numpy(dtype=float)
        src_lon = src_coords[config.lon_col].to_numpy(dtype=float)
        dst_lat = dst_coords[config.lat_col].to_numpy(dtype=float)
        dst_lon = dst_coords[config.lon_col].to_numpy(dtype=float)

        hav_m = _haversine_pairwise_m(src_lat, src_lon, dst_lat, dst_lon)
        comp_bearing = _bearing_pairwise_deg(src_lat, src_lon, dst_lat, dst_lon)

        # Fill only what is missing; preserve lookup-provided values when present
        if np.any(needs_distance):
            final_distance_m = final_distance_m.copy()
            final_distance_m[needs_distance] = hav_m[needs_distance] * stretch
        if np.any(needs_bearing):
            bearing_deg = bearing_deg.copy()
            bearing_deg[needs_bearing] = comp_bearing[needs_bearing]

        # Bins depend on the (possibly filled) continuous values
        if np.any(needs_dist_bin):
            distance_bin = distance_bin.copy()
            distance_bin[needs_dist_bin] = _bin_distances_m(
                np.asarray(final_distance_m, dtype=float)[needs_dist_bin],
                edges_m=dist_edges_m,
            )
        if np.any(needs_dir_bin):
            direction_bin = direction_bin.copy()
            direction_bin[needs_dir_bin] = _bearing_deg_to_direction_bin(
                np.asarray(bearing_deg, dtype=float)[needs_dir_bin]
            )

    return pd.DataFrame(
        {
            config.session_id_col: session_id,
            "transition_index": transition_index,
            "src_POIId": src_poi,
            "dst_POIId": dst_poi,
            "gap_s": gap_s,
            "gap_bin": gap_bin,
            "final_distance_m": final_distance_m,
            "distance_bin": distance_bin,
            "bearing_deg": bearing_deg,
            "direction_bin": direction_bin,
        },
        columns=output_cols,
    )


# ---------------------------------------------------------------------------
# Batch: all sessions
# ---------------------------------------------------------------------------


def build_all_session_transition_descriptors(
    checkins_df: pd.DataFrame,
    pair_lookup: dict[tuple, dict],
    poi_coord_map: dict,
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
        config.timestamp_col,
        config.poi_id_col,
    ]
    missing = [c for c in required_cols if c not in checkins_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in checkins_df: {missing}")

    df = checkins_df.copy()
    df[config.timestamp_col] = pd.to_datetime(df[config.timestamp_col], errors="coerce")
    df = df.sort_values([config.session_id_col, config.timestamp_col, config.poi_id_col]).reset_index(
        drop=True
    )

    groups = df.groupby(config.session_id_col, sort=False)
    iterator = tqdm(groups, desc="Computing transitions", unit="session") if show_progress else groups

    lookup_df = pd.DataFrame.from_dict(pair_lookup, orient="index")
    lookup_df.index = pd.MultiIndex.from_tuples(lookup_df.index, names=["src_POIId", "dst_POIId"])  # fmt: skip
    coord_df = pd.DataFrame.from_dict(poi_coord_map, orient="index")

    all_transitions: list[pd.DataFrame] = []
    for session_id, session_df in iterator:
        t = compute_single_session_transitions(
            session_df=session_df,
            lookup_df=lookup_df,
            coord_df=coord_df,
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


if __name__ == "__main__":
    config = SpatialEncodingConfig()

    city = "tky"
    scrip_dir = Path(__file__).resolve().parent.parent

    cprint(f"\nLoading {city} raw checkins data...", "yellow")
    checkins_df = pd.read_csv(scrip_dir / f"data/{city}/train_sample.csv")
    checkins_df = checkins_df.rename(columns={"pseudo_session_trajectory_id": "SessionId"})

    poi_df = pd.read_csv(scrip_dir / f"artifacts/{city}/{city}_poi.csv")
    pair_lookup_df = pd.read_csv(scrip_dir / f"artifacts/{city}/{city}_poi_pair_lookup_table.csv")

    pair_lookup = build_pair_lookup_dict(pair_lookup_df)
    poi_coord_map = build_poi_coord_map(poi_df, config)

    session_transition_df = build_all_session_transition_descriptors(
        checkins_df, pair_lookup, poi_coord_map, config
    )
    cache_path = scrip_dir / f"artifacts/{city}/{city}_session_transition.csv"
    session_transition_df.to_csv(cache_path)

    with open(scrip_dir / f"artifacts/{city}/{city}_pair_lookup.pkl", "wb") as f:
        pickle.dump(pair_lookup, f)

    with open(scrip_dir / f"artifacts/{city}/{city}_poi_coord_map.pkl", "wb") as f:
        pickle.dump(poi_coord_map, f)
