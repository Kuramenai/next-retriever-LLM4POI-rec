from __future__ import annotations

# from typing import Optional
# import math

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import BallTree

from extract_poi_spatial_descriptors import (
    _make_poi_gdf,
    _map_pois_to_nearest_graph_nodes,
    _get_undirected_graph,
)

EARTH_RADIUS_M = 6_371_008.8


def _haversine_from_one_to_many_m(
    lat1: float,
    lon1: float,
    lats2: np.ndarray,
    lons2: np.ndarray,
) -> np.ndarray:
    """
    Vectorized haversine distance from one point to many points, in meters.
    """
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lats2_rad = np.radians(lats2)
    lons2_rad = np.radians(lons2)

    dlat = lats2_rad - lat1_rad
    dlon = lons2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lats2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


def _bearing_from_one_to_many_deg(
    lat1: float,
    lon1: float,
    lats2: np.ndarray,
    lons2: np.ndarray,
) -> np.ndarray:
    """
    Vectorized initial bearing from one point to many points, in degrees [0, 360).
    """
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lats2_rad = np.radians(lats2)
    lons2_rad = np.radians(lons2)

    dlon = lons2_rad - lon1_rad

    y = np.sin(dlon) * np.cos(lats2_rad)
    x = np.cos(lat1_rad) * np.sin(lats2_rad) - np.sin(lat1_rad) * np.cos(
        lats2_rad
    ) * np.cos(dlon)

    bearings = np.degrees(np.arctan2(y, x))
    return (bearings + 360.0) % 360.0


def _bearing_deg_to_direction_bin(bearings_deg: np.ndarray) -> np.ndarray:
    """
    Map bearings to 8-way cardinal/intercardinal bins.
    """
    labels = np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], dtype=object)
    idx = (((bearings_deg + 22.5) % 360.0) // 45.0).astype(int)
    return labels[idx]


def _distance_bin_labels(edges_m: tuple[float, ...]) -> list[str]:
    labels = []
    lower = 0
    for edge in edges_m:
        labels.append(f"{int(lower)}-{int(edge)}m")
        lower = edge
    labels.append(f"{int(edges_m[-1])}+m")
    return labels


def _bin_distances_m(
    distances_m: np.ndarray,
    edges_m: tuple[float, ...],
) -> np.ndarray:
    """
    Bin distances using fixed absolute meter thresholds.
    """
    bins = [0.0, *edges_m, np.inf]
    labels = _distance_bin_labels(edges_m)

    return (
        pd.cut(
            distances_m,
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=True,
        )
        .astype(object)
        .to_numpy()
    )


def build_sparse_pair_transition_lookup(
    poi_df: pd.DataFrame,
    road_graph,
    config,
    *,
    force_recompute_nearest_nodes: bool = False,
) -> pd.DataFrame:
    """
    Build a sparse pairwise transition lookup over top-k nearby POIs.

    Expected columns in poi_df
    --------------------------
    - config.poi_id_col
    - config.lat_col
    - config.lon_col
    - optionally 'nearest_graph_node_id'
      (if missing, it will be computed)

    Output columns
    --------------
    - src_POIId
    - dst_POIId
    - src_node_id
    - dst_node_id
    - haversine_distance_m
    - road_distance_m
    - used_haversine_fallback
    - bearing_deg
    - direction_bin
    - distance_bin
    """
    required_cols = [config.poi_id_col, config.lat_col, config.lon_col]
    missing = [c for c in required_cols if c not in poi_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in poi_df: {missing}")

    df = poi_df.copy().reset_index(drop=True)

    # Ensure nearest graph node is present
    if ("nearest_graph_node_id" not in df.columns) or force_recompute_nearest_nodes:
        poi_gdf = _make_poi_gdf(df, config)
        df["nearest_graph_node_id"] = _map_pois_to_nearest_graph_nodes(
            poi_gdf, road_graph
        )

    # Use an undirected graph for local path lengths
    G_work = _get_undirected_graph(road_graph)

    # BallTree on lat/lon radians for top-k Haversine neighbors
    coords_deg = df[[config.lat_col, config.lon_col]].to_numpy(dtype=float)
    coords_rad = np.radians(coords_deg)

    tree = BallTree(coords_rad, metric="haversine")

    k = min(config.topk_pair_neighbors + 1, len(df))
    dist_rad, neighbor_idx = tree.query(coords_rad, k=k)
    dist_m = dist_rad * EARTH_RADIUS_M

    poi_ids = df[config.poi_id_col].to_numpy()
    lats = df[config.lat_col].to_numpy(dtype=float)
    lons = df[config.lon_col].to_numpy(dtype=float)
    node_ids = df["nearest_graph_node_id"].to_numpy()

    records = []

    for i in range(len(df)):
        src_poi_id = poi_ids[i]
        src_lat = lats[i]
        src_lon = lons[i]
        src_node = node_ids[i]

        # Exclude self neighbor at position 0
        tgt_idx = neighbor_idx[i, 1:]
        tgt_haversine_m = dist_m[i, 1:]

        if len(tgt_idx) == 0:
            continue

        tgt_poi_ids = poi_ids[tgt_idx]
        tgt_lats = lats[tgt_idx]
        tgt_lons = lons[tgt_idx]
        tgt_nodes = node_ids[tgt_idx]

        # Bearing + direction from coordinates
        bearings_deg = _bearing_from_one_to_many_deg(
            src_lat, src_lon, tgt_lats, tgt_lons
        )
        direction_bins = _bearing_deg_to_direction_bin(bearings_deg)

        # Local Dijkstra cutoff based on farthest neighbor in this sparse neighborhood
        # The stretch factor gives the road network some slack over straight-line distance.
        max_hav = float(np.nanmax(tgt_haversine_m))
        cutoff_m = max_hav * float(config.road_distance_stretch_factor) + 50.0

        # Compute local shortest-path distances from source node
        if src_node in G_work:
            try:
                sp_lengths = nx.single_source_dijkstra_path_length(
                    G_work,
                    source=src_node,
                    cutoff=cutoff_m,
                    weight="length",
                )
            except Exception:
                sp_lengths = {}
        else:
            sp_lengths = {}

        road_distances_m = np.array(
            [sp_lengths.get(tgt_node, np.nan) for tgt_node in tgt_nodes],
            dtype=float,
        )

        used_haversine_fallback = np.isnan(road_distances_m)
        final_distance_m = np.where(
            used_haversine_fallback,
            tgt_haversine_m,
            road_distances_m,
        )

        distance_bins = _bin_distances_m(
            final_distance_m,
            edges_m=tuple(config.distance_bin_edges_m),
        )

        for j in range(len(tgt_idx)):
            records.append(
                {
                    "src_POIId": src_poi_id,
                    "dst_POIId": tgt_poi_ids[j],
                    "src_node_id": src_node,
                    "dst_node_id": tgt_nodes[j],
                    "haversine_distance_m": float(tgt_haversine_m[j]),
                    "road_distance_m": (
                        float(road_distances_m[j])
                        if not np.isnan(road_distances_m[j])
                        else np.nan
                    ),
                    "used_haversine_fallback": bool(used_haversine_fallback[j]),
                    "bearing_deg": float(bearings_deg[j]),
                    "direction_bin": direction_bins[j],
                    "distance_bin": distance_bins[j],
                }
            )

    pair_df = pd.DataFrame.from_records(records)

    # Defensive check: src/dst should be unique in this sparse lookup
    dup_mask = pair_df.duplicated(subset=["src_POIId", "dst_POIId"])
    if dup_mask.any():
        pair_df = pair_df.loc[~dup_mask].copy()

    return pair_df
