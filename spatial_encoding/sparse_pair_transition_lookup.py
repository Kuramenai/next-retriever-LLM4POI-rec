from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from sklearn.neighbors import BallTree
import concurrent.futures
import multiprocessing

from tqdm import tqdm
from termcolor import cprint
from spatial_encoding.extract_poi_spatial_descriptors import SpatialEncodingConfig

from spatial_encoding.extract_poi_spatial_descriptors import (
    _make_poi_gdf,
    _map_pois_to_nearest_graph_nodes,
    _get_undirected_graph,
)

EARTH_RADIUS_M = 6_371_008.8

# Global variable for multiprocessing to avoid passing the massive road graph via pickle
_GLOBAL_ROAD_GRAPH = None


def _init_worker(graph_data):
    """Initializer for process pool to share the road graph in memory."""
    global _GLOBAL_ROAD_GRAPH
    _GLOBAL_ROAD_GRAPH = graph_data


def _compute_dijkstra_chunk(src_nodes: list[int], cutoff_m: float) -> dict[int, dict]:
    """Worker function to compute shortest paths for a chunk of source nodes."""
    results = {}
    for src_node in src_nodes:
        if src_node in _GLOBAL_ROAD_GRAPH:
            try:
                sp_lengths = nx.single_source_dijkstra_path_length(
                    _GLOBAL_ROAD_GRAPH,
                    source=src_node,
                    cutoff=cutoff_m,
                    weight="length",
                )
                results[src_node] = sp_lengths
            except Exception:
                results[src_node] = {}
        else:
            results[src_node] = {}
    return results


def _haversine_from_one_to_many_m(
    lat1: float, lon1: float, lats2: np.ndarray, lons2: np.ndarray
) -> np.ndarray:
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
    lat1: float, lon1: float, lats2: np.ndarray, lons2: np.ndarray
) -> np.ndarray:
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


def _bin_distances_m(distances_m: np.ndarray, edges_m: tuple[float, ...]) -> np.ndarray:
    bins = [0.0, *edges_m, np.inf]
    labels = _distance_bin_labels(edges_m)
    return pd.cut(
        distances_m, bins=bins, labels=labels, include_lowest=True, right=True
    ).astype(object)


def build_sparse_pair_transition_lookup(
    poi_df: pd.DataFrame,
    road_graph,
    config,
    *,
    force_recompute_nearest_nodes: bool = False,
    max_radius_m: float = 1000.0,
    dijkstra_cutoff_padding_m: float = 50.0,
) -> pd.DataFrame:
    """
    Build a sparse pairwise transition lookup over nearby POIs within a strict radius.

    Note on Transit: Because the target cities (NYC/Tokyo) rely heavily on transit,
    ensure `config.distance_bin_edges_m` extends to larger bounds (e.g., 3000, 5000, 10000)
    to capture subway jumps, and ensure the road_graph is a pedestrian ('walk') graph.
    """
    required_cols = [config.poi_id_col, config.lat_col, config.lon_col]
    if missing := [c for c in required_cols if c not in poi_df.columns]:
        raise ValueError(f"Missing required columns in poi_df: {missing}")

    df = poi_df.copy().reset_index(drop=True)

    # fmt: off
    if ("nearest_graph_node_id" not in df.columns) or force_recompute_nearest_nodes:
        poi_gdf = _make_poi_gdf(df, config)
        df["nearest_graph_node_id"] = _map_pois_to_nearest_graph_nodes(poi_gdf, road_graph)

    # Use undirected graph to simulate pedestrian freedom (crosswalks, parks)
    G_work = _get_undirected_graph(road_graph)

    coords_deg = df[[config.lat_col, config.lon_col]].to_numpy(dtype=float)
    coords_rad = np.radians(coords_deg)
    tree = BallTree(coords_rad, metric="haversine")

    # LOGICAL FIX: Use strict radius (e.g., 3km) instead of rigid Top-K
    radius_m = max_radius_m
    radius_rad = radius_m / EARTH_RADIUS_M
    neighbor_indices_list, neighbor_dists_rad_list = tree.query_radius(
        coords_rad, r=radius_rad, return_distance=True
    )

    poi_ids = df[config.poi_id_col].to_numpy()
    lats = df[config.lat_col].to_numpy(dtype=float)
    lons = df[config.lon_col].to_numpy(dtype=float)
    node_ids = df["nearest_graph_node_id"].to_numpy()

    # MULTIPROCESSING: Chunk source nodes and compute Dijkstra in parallel
    unique_src_nodes = np.unique(node_ids)
    chunk_size = max(1, len(unique_src_nodes) // multiprocessing.cpu_count())
    node_chunks = [unique_src_nodes[i : i + chunk_size] for i in range(0, len(unique_src_nodes), chunk_size)]
    chunk_sizes = [len(chunk) for chunk in node_chunks]
    total_nodes = sum(chunk_sizes)

    cutoff_m = radius_m * float(config.road_distance_stretch_factor) + dijkstra_cutoff_padding_m
    sp_lengths_master = {}

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=multiprocessing.cpu_count(),
        initializer=_init_worker,
        initargs=(G_work,),
    ) as executor:
        futures = [executor.submit(_compute_dijkstra_chunk, chunk, cutoff_m) for chunk in node_chunks]
        
        # Map each future → its chunk size
        future_to_size = {
        future: size for future, size in zip(futures, chunk_sizes)}
        with tqdm(total=total_nodes, desc="Processing nodes", unit="node") as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                sp_lengths_master.update(result)
                pbar.update(future_to_size[future])

    # SCALABILITY FIX: Columnar allocation prevents OOM from millions of dictionaries
    cols = {
        "src_POIId": [],
        "dst_POIId": [],
        "src_node_id": [],
        "dst_node_id": [],
        "haversine_distance_m": [],
        "road_distance_m": [],
        "used_haversine_fallback": [],
        "bearing_deg": [],
        "direction_bin": [],
        "distance_bin": [],
        "distance_source": [],
        "final_distance_m": [],
    }

    # fmt:off

    for i in tqdm(range(len(df)), total=len(df), desc="Calculating pair"):
        # Filter out self-loops (distance == 0)
        mask = neighbor_dists_rad_list[i] > 0
        tgt_idx = neighbor_indices_list[i][mask]
        tgt_haversine_m = neighbor_dists_rad_list[i][mask] * EARTH_RADIUS_M

        if len(tgt_idx) == 0:
            continue

        tgt_poi_ids = poi_ids[tgt_idx]
        tgt_nodes = node_ids[tgt_idx]

        bearings_deg = _bearing_from_one_to_many_deg(lats[i], lons[i], lats[tgt_idx], lons[tgt_idx])
        direction_bins = _bearing_deg_to_direction_bin(bearings_deg)

        sp_lengths = sp_lengths_master.get(node_ids[i], {})
        road_distances_m = np.array([sp_lengths.get(t_node, np.nan) for t_node in tgt_nodes], dtype=float)

        used_haversine_fallback = np.isnan(road_distances_m)

        # LOGICAL FIX: Scale the Haversine fallback to mimic road network penalties
        scaled_haversine = tgt_haversine_m * float(config.road_distance_stretch_factor)
        final_distance_m = np.where(used_haversine_fallback, scaled_haversine, road_distances_m)
        distance_bins = _bin_distances_m(final_distance_m, edges_m=tuple(config.distance_bin_edges_m))

        # Append to flat columnar lists
        cols["src_POIId"].extend(np.repeat(poi_ids[i], len(tgt_idx)))
        cols["dst_POIId"].extend(tgt_poi_ids)
        cols["src_node_id"].extend(np.repeat(node_ids[i], len(tgt_idx)))
        cols["dst_node_id"].extend(tgt_nodes)
        cols["distance_source"].extend(np.where(used_haversine_fallback, "scaled_haversine", "graph").tolist())
        cols["haversine_distance_m"].extend(tgt_haversine_m)
        cols["road_distance_m"].extend(road_distances_m)
        cols["used_haversine_fallback"].extend(used_haversine_fallback)
        cols["bearing_deg"].extend(bearings_deg)
        cols["direction_bin"].extend(direction_bins)
        cols["distance_bin"].extend(distance_bins)
        cols["final_distance_m"].extend(final_distance_m)

    pair_df = pd.DataFrame(cols)

    dup_mask = pair_df.duplicated(subset=["src_POIId", "dst_POIId"])
    if dup_mask.any():
        pair_df = pair_df.loc[~dup_mask].copy()

    return pair_df


if __name__ == "__main__":
    config = SpatialEncodingConfig()

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

    pair_df = build_sparse_pair_transition_lookup(poi_df, road_graph, config)

    cache_path = scrip_dir / f"artifacts/{city}/{city}_poi_pair_lookup_table.csv"

    pair_df.to_csv(cache_path)
