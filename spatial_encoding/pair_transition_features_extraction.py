from __future__ import annotations

import numpy as np
import pandas as pd

import pickle
from pathlib import Path 
from termcolor import cprint
from spatial_encoding.sparse_pair_transition_lookup import  _bin_distances_m, _bearing_deg_to_direction_bin, SpatialEncodingConfig


EARTH_RADIUS_M = 6_371_000.0


def _temporal_gap_bin_labels(edges_s: tuple[float, ...]) -> list[str]:
    labels = []
    lower = 0.0
    for edge in edges_s:
        labels.append(f"{int(lower)}-{int(edge)}s")
        lower = edge
    labels.append(f"{int(edges_s[-1])}+s")
    return labels


def _bin_temporal_gaps_s(gaps_s: np.ndarray, edges_s: tuple[float, ...]) -> np.ndarray:
    bins = [0.0, *edges_s, np.inf]
    labels = _temporal_gap_bin_labels(edges_s)
    return (
        pd.cut(
            gaps_s,
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=True,
        )
        .astype(object))


def _resolve_temporal_gap_edges_seconds(config) -> tuple[float, ...]:
    if hasattr(config, "temporal_gap_bin_edges_seconds"):
        return tuple(float(x) for x in config.temporal_gap_bin_edges_seconds)

    if hasattr(config, "temporal_gap_bin_edges_minutes"):
        return tuple(float(x) * 60.0 for x in config.temporal_gap_bin_edges_minutes)

    if hasattr(config, "temporal_gap_bin_edges_hours"):
        return tuple(float(x) * 3600.0 for x in config.temporal_gap_bin_edges_hours)

    # default: 15 min, 30 min, 1 h, 2 h, 6 h
    return (900.0, 1800.0, 3600.0, 7200.0, 21600.0)


def _ensure_pair_lookup_final_distance(
    pair_lookup_df: pd.DataFrame,
    road_distance_stretch_factor: float,
) -> pd.DataFrame:
    df = pair_lookup_df.copy()

    if "final_distance_m" in df.columns:
        return df

    if {
        "road_distance_m",
        "haversine_distance_m",
        "used_haversine_fallbackk",
    }.issubset(df.columns):
        df["final_distance_m"] = np.where(
            df["used_haversine_fallback"].astype(bool),
            df["haversine_distance_m"].astype(float) * float(road_distance_stretch_factor),
            df["road_distance_m"].astype(float),
        )
        return df

    if "road_distance_m" in df.columns:
        df["final_distance_m"] = df["road_distance_m"].astype(float)
        return df

    raise ValueError(
        "pair_lookup_df must contain final_distance_m or enough columns "
        "to reconstruct it."
    )


def _pairwise_haversine_and_bearing_m_deg(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lat1_rad = np.radians(lat1.astype(float))
    lon1_rad = np.radians(lon1.astype(float))
    lat2_rad = np.radians(lat2.astype(float))
    lon2_rad = np.radians(lon2.astype(float))

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    distance_m = EARTH_RADIUS_M * c

    y = np.sin(dlon) * np.cos(lat2_rad)
    x = (
        np.cos(lat1_rad) * np.sin(lat2_rad)
        - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    )
    bearing_deg = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0

    return distance_m, bearing_deg


def build_observed_session_transition_descriptors(
    checkins_df: pd.DataFrame,
    pair_lookup_df: pd.DataFrame,
    poi_df: pd.DataFrame,
    config,
    *,
    allow_offlookup_fallback: bool = True,
    sort_output: bool = True,
) -> pd.DataFrame:
    """
    Build one row per observed consecutive transition inside each session.

    Output includes:
      - temporal_gap_s / temporal_gap_bin
      - final_distance_m / distance_bin
      - bearing_deg / direction_bin
      - distance_source
      - found_in_pair_lookup

    Notes
    -----
    - This is the roadmap-aligned per-transition artifact.
    - It consumes the static sparse pair lookup for distance/direction.
    - Temporal gap is always computed from actual session timestamps here.
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

    required_poi_cols = [
        config.poi_id_col,
        config.lat_col,
        config.lon_col,
    ]
    missing_poi = [c for c in required_poi_cols if c not in poi_df.columns]
    if missing_poi:
        raise ValueError(f"Missing required columns in poi_df: {missing_poi}")

    # ------------------------------------------------------------------
    # Normalize timestamps and sort chronologically within sessions
    # ------------------------------------------------------------------
    df = checkins_df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df[config.timestamp_col]):
        df[config.timestamp_col] = pd.to_datetime(df[config.timestamp_col], errors="coerce")

    if df[config.timestamp_col].isna().any():
        bad_count = int(df[config.timestamp_col].isna().sum())
        raise ValueError(f"{bad_count} rows have invalid timestamps after parsing.")

    sort_cols = [config.session_id_col, config.timestamp_col]
    if hasattr(config, "user_id_col") and config.user_id_col in df.columns:
        sort_cols = [config.user_id_col, *sort_cols]

    df = df.sort_values(sort_cols).reset_index(drop=True)
    df["checkin_index_in_session"] = df.groupby(config.session_id_col).cumcount()

    # ------------------------------------------------------------------
    # Build consecutive transitions
    # ------------------------------------------------------------------
    g = df.groupby(config.session_id_col, sort=False)

    df["dst_POIId"] = g[config.poi_id_col].shift(-1)
    df["dst_timestamp"] = g[config.timestamp_col].shift(-1)

    transition_df = df.loc[df["dst_POIId"].notna()].copy()

    transition_df = transition_df.rename(
        columns={
            config.poi_id_col: "src_POIId",
            config.timestamp_col: "src_timestamp",
        }
    )

    transition_df["transition_index"] = transition_df["checkin_index_in_session"]

    transition_df["temporal_gap_s"] = (
        transition_df["dst_timestamp"] - transition_df["src_timestamp"]
    ).dt.total_seconds()

    if (transition_df["temporal_gap_s"] < 0).any():
        raise ValueError("Negative temporal gaps found after sorting. Check session ordering.")

    gap_edges_s = _resolve_temporal_gap_edges_seconds(config)
    transition_df["temporal_gap_bin"] = _bin_temporal_gaps_s(
        transition_df["temporal_gap_s"].to_numpy(dtype=float),
        edges_s=gap_edges_s,
    )

    # ------------------------------------------------------------------
    # Attach static pairwise spatial descriptors from sparse lookup
    # ------------------------------------------------------------------
    pair_lookup = _ensure_pair_lookup_final_distance(
        pair_lookup_df=pair_lookup_df,
        road_distance_stretch_factor=float(config.road_distance_stretch_factor),
    )

    pair_required = ["src_POIId", "dst_POIId"]
    missing_pair = [c for c in pair_required if c not in pair_lookup.columns]
    if missing_pair:
        raise ValueError(f"Missing required columns in pair_lookup_df: {missing_pair}")

    keep_pair_cols = ["src_POIId", "dst_POIId"]
    for col in [
        "final_distance_m",
        "distance_bin",
        "bearing_deg",
        "direction_bin",
        "distance_source",
        "road_distance_m",
        "haversine_distance_m",
        "used_haversine_fallback",
    ]:
        if col in pair_lookup.columns:
            keep_pair_cols.append(col)

    pair_lookup = pair_lookup[keep_pair_cols].drop_duplicates(
        subset=["src_POIId", "dst_POIId"],
        keep="first",
    )

    transition_df = transition_df.merge(
        pair_lookup,
        on=["src_POIId", "dst_POIId"],
        how="left",
    )

    transition_df["found_in_pair_lookup"] = transition_df["final_distance_m"].notna()

    # ------------------------------------------------------------------
    # Off-lookup fallback: scaled haversine + bearing, computed only for misses
    # ------------------------------------------------------------------
    if allow_offlookup_fallback:
        missing_mask = ~transition_df["found_in_pair_lookup"]

        if missing_mask.any():
            poi_coord_map = (
                poi_df[[config.poi_id_col, config.lat_col, config.lon_col]]
                .drop_duplicates(subset=[config.poi_id_col], keep="first")
                .set_index(config.poi_id_col)[[config.lat_col, config.lon_col]]
            )

            miss_src_ids = transition_df.loc[missing_mask, "src_POIId"].to_numpy()
            miss_dst_ids = transition_df.loc[missing_mask, "dst_POIId"].to_numpy()

            src_coords = poi_coord_map.reindex(miss_src_ids)
            dst_coords = poi_coord_map.reindex(miss_dst_ids)

            if src_coords.isna().any().any() or dst_coords.isna().any().any():
                raise ValueError(
                    "Missing POI coordinates for some off-lookup transitions. "
                    "Check that all src/dst POIs exist in poi_df."
                )

            fallback_haversine_m, fallback_bearing_deg = _pairwise_haversine_and_bearing_m_deg(
                lat1=src_coords[config.lat_col].to_numpy(),
                lon1=src_coords[config.lon_col].to_numpy(),
                lat2=dst_coords[config.lat_col].to_numpy(),
                lon2=dst_coords[config.lon_col].to_numpy(),
            )

            fallback_final_distance_m = (
                fallback_haversine_m * float(config.road_distance_stretch_factor)
            )
            fallback_direction_bin = _bearing_deg_to_direction_bin(fallback_bearing_deg)
            fallback_distance_bin = _bin_distances_m(
                fallback_final_distance_m,
                edges_m=tuple(config.distance_bin_edges_m),
            )

            miss_idx = transition_df.index[missing_mask]

            for col, default in [
                ("haversine_distance_m", np.nan),
                ("road_distance_m", np.nan),
                ("used_haversine_fallback", False),
                ("distance_source", np.nan),
                ("final_distance_m", np.nan),
                ("bearing_deg", np.nan),
                ("direction_bin", np.nan),
                ("distance_bin", np.nan),
            ]:
                if col not in transition_df.columns:
                    transition_df[col] = default

            transition_df.loc[miss_idx, "haversine_distance_m"] = fallback_haversine_m
            transition_df.loc[miss_idx, "road_distance_m"] = np.nan
            transition_df.loc[miss_idx, "final_distance_m"] = fallback_final_distance_m
            transition_df.loc[miss_idx, "bearing_deg"] = fallback_bearing_deg
            transition_df.loc[miss_idx, "direction_bin"] = fallback_direction_bin
            transition_df.loc[miss_idx, "distance_bin"] = fallback_distance_bin
            transition_df.loc[miss_idx, "used_haversine_fallback"] = True
            transition_df.loc[miss_idx, "distance_source"] = "offlookup_scaled_haversine"

    # ------------------------------------------------------------------
    # Final ordering
    # ------------------------------------------------------------------
    preferred_cols = []

    if hasattr(config, "user_id_col") and config.user_id_col in transition_df.columns:
        preferred_cols.append(config.user_id_col)

    preferred_cols += [
        config.session_id_col,
        "transition_index",
        "checkin_index_in_session",
        "src_POIId",
        "dst_POIId",
        "src_timestamp",
        "dst_timestamp",
        "temporal_gap_s",
        "temporal_gap_bin",
        "final_distance_m",
        "distance_bin",
        "bearing_deg",
        "direction_bin",
        "distance_source",
        "found_in_pair_lookup",
    ]

    for extra_col in [
        "haversine_distance_m",
        "road_distance_m",
        "used_haversine_fallback",
    ]:
        if extra_col in transition_df.columns:
            preferred_cols.append(extra_col)

    existing_cols = [c for c in preferred_cols if c in transition_df.columns]
    remaining_cols = [c for c in transition_df.columns if c not in existing_cols]

    transition_df = transition_df[existing_cols + remaining_cols]

    if sort_output:
        sort_cols = [config.session_id_col, "transition_index"]
        if hasattr(config, "user_id_col") and config.user_id_col in transition_df.columns:
            sort_cols = [config.user_id_col, *sort_cols]
        transition_df = transition_df.sort_values(sort_cols).reset_index(drop=True)

    return transition_df