from __future__ import annotations

from dataclasses import dataclass

# from os import name
from typing import Mapping, Optional, Sequence
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox

# from shapely.geometry import Point
import h3
from scipy.spatial import cKDTree
from termcolor import cprint

# from pyrosm import OSM


import pickle


@dataclass
class SpatialEncodingConfig:
    h3_res_coarse: int = 8
    h3_res_fine: int = 9

    density_radius_m: float = 100.0
    density_num_bins: int = 5
    connectivity_num_bins : int = 5

    # Column names in the incoming POI dataframe
    poi_id_col: str = "PId"
    lat_col: str = "Latitude"
    lon_col: str = "Longitude"

    # Output token prefixes
    density_prefix: str = "D"
    connectivity_prefix:str = "C"
    
    road_distance_stretch_factor: float = 2.0
    distance_bin_edges_m: tuple = (250, 500, 1000, 2000, 5000)

    session_id_col: str = "SessionId"
    timestamp_col: str = "UTCTimeOffset"
    category_col: str = "Category"
    gap_bin_edges_min: tuple = (15, 30, 60, 120, 240)
    
    

def _validate_poi_columns(df: pd.DataFrame, config: SpatialEncodingConfig) -> None:
    required = [config.poi_id_col, config.lat_col, config.lon_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required POI columns: {missing}")


def _make_poi_gdf(df: pd.DataFrame, config: SpatialEncodingConfig) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[config.lon_col], df[config.lat_col]),
        crs="EPSG:4326",
    )
    return gdf


def _project_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    projected_crs = gdf.estimate_utm_crs()
    if projected_crs is None:
        raise ValueError("Could not estimate a projected CRS for POI geometries.")
    return gdf.to_crs(projected_crs)


def _rank_bin(
    values: pd.Series,
    prefix: str,
    num_bins: int,
    unknown_token: Optional[str] = None,
) -> pd.Series:
    """
    Robust quantile-style binning based on percentile rank.
    This is safer than qcut when many ties exist.
    """
    if unknown_token is None:
        unknown_token = f"{prefix}_UNK"

    s = pd.Series(values, index=values.index, dtype="float64")
    out = pd.Series(unknown_token, index=s.index, dtype="object")

    valid_mask = s.notna()
    if valid_mask.sum() == 0:
        return out

    ranked = s.loc[valid_mask].rank(method="average", pct=True)
    bin_ids = np.ceil(ranked * num_bins).clip(1, num_bins).astype(int)
    out.loc[valid_mask] = [f"{prefix}{b}" for b in bin_ids]

    return out


def _compute_density_counts(
    poi_proj_gdf: gpd.GeoDataFrame,
    radius_m: float,
) -> pd.Series:
    """
    Count number of POIs within radius_m of each POI, excluding self.
    """
    # fmt: off
    coords = np.column_stack([poi_proj_gdf.geometry.x.values, poi_proj_gdf.geometry.y.values])
    tree = cKDTree(coords)

    neighbor_indices = tree.query_ball_point(coords, r=radius_m)
    counts = np.array([max(len(idx_list) - 1, 0) for idx_list in neighbor_indices], dtype=int)

    return pd.Series(counts, index=poi_proj_gdf.index, name="density_count")


def _get_graph_crs(road_graph) -> str:
    graph_crs = road_graph.graph.get("crs", None)
    if graph_crs is None:
        # Most OSMnx graphs carry a CRS, but default to WGS84 if absent
        graph_crs = "EPSG:4326"
    return graph_crs


def _map_pois_to_nearest_graph_nodes(
    poi_gdf_wgs84: gpd.GeoDataFrame,
    road_graph,
) -> pd.Series:
    """
    Map each POI to its nearest road-graph node using the graph CRS.
    """
    graph_crs = _get_graph_crs(road_graph)
    poi_for_graph = poi_gdf_wgs84.to_crs(graph_crs)

    xs = poi_for_graph.geometry.x.to_numpy()
    ys = poi_for_graph.geometry.y.to_numpy()

    nearest_nodes = ox.distance.nearest_nodes(road_graph, X=xs, Y=ys)
    return pd.Series(
        nearest_nodes, index=poi_gdf_wgs84.index, name="nearest_graph_node_id"
    )


def _get_undirected_graph(road_graph) -> nx.Graph:
    """
    Convert an OSMnx road graph to an undirected simple graph.
    """
    try:
        G_u = ox.convert.to_undirected(road_graph)
    except Exception:
        G_u = road_graph.to_undirected()

    if not isinstance(G_u, nx.Graph):
        G_u = nx.Graph(G_u)

    return G_u


def _graph_cache_signature(road_graph) -> dict:
    """
    Lightweight signature for cache validation.
    This is not cryptographic; it is just a practical stale-cache guard.
    """
    return {
        "num_nodes": int(road_graph.number_of_nodes()),
        "num_edges": int(road_graph.number_of_edges()),
        "crs": str(road_graph.graph.get("crs", "UNKNOWN")),
        "name": str(road_graph.graph.get("name", "UNKNOWN")),
    }





def build_poi_spatial_descriptors(
    poi_df: pd.DataFrame,
    road_graph,
    config: SpatialEncodingConfig,
) -> pd.DataFrame:
    """
    Build per-POI spatial descriptors for Module 2.

    Parameters
    ----------
    poi_df:
        DataFrame with at least POI ID, latitude, longitude.
    road_graph:
        OSMnx road graph.
    config:
        Spatial encoding configuration.
    context_gdf:
        Optional GeoDataFrame for nearby OSM context features.
        Only used if config.use_urban_context=True.

    Returns
    -------
    descriptor_df:
        DataFrame keyed by POI ID with region, density
        nearest node, and optional urban context descriptors.
    """
    _validate_poi_columns(poi_df, config)

    # 1) Base POI GeoDataFrames
    cprint(
        "\nProjecting POIs longitudes, latitudes values to new Coordinate Reference System (CRS)...",
        "yellow",
    )
    poi_gdf_wgs84 = _make_poi_gdf(poi_df, config)
    poi_proj_gdf = _project_gdf(poi_gdf_wgs84)
    cprint("Projection Complete.", "green")

    # 2) Multi-scale H3 region tokens
    cprint("\nH3 region assignment...", "yellow")
    descriptor_df = poi_df.copy()

    descriptor_df["region_coarse_token"] = poi_gdf_wgs84.apply(
        lambda row: h3.latlng_to_cell(
            row[config.lat_col], row[config.lon_col], config.h3_res_coarse
        ),
        axis=1,
    )
    descriptor_df["region_fine_token"] = poi_gdf_wgs84.apply(
        lambda row: h3.latlng_to_cell(
            row[config.lat_col], row[config.lon_col], config.h3_res_fine
        ),
        axis=1,
    )
    cprint("Assignment done.", "green")

    # 3) Density counts + bins
    cprint("\nComputing POIs density...", "yellow")
    density_counts = _compute_density_counts(
        poi_proj_gdf, radius_m=config.density_radius_m
    )
    descriptor_df["density_count"] = density_counts.values
    descriptor_df["density_bin"] = _rank_bin(
        density_counts,
        prefix=config.density_prefix,
        num_bins=config.density_num_bins,
    ).values
    cprint("POIs density computation is complete.", "green")

    # 4) Nearest road-graph node
    cprint("\nGet POIs nearest road-graph nodes...", "yellow")
    nearest_nodes = _map_pois_to_nearest_graph_nodes(poi_gdf_wgs84, road_graph)
    descriptor_df["nearest_graph_node_id"] = nearest_nodes.values
    cprint("POIs nearest road-graph node acquired.", "green")
    
    degree = dict(road_graph.degree(weight=None))
    descriptor_df["node_degree"] = nearest_nodes.map(degree)
    descriptor_df["connectivity_bin"] = _rank_bin(descriptor_df["node_degree"], config.connectivity_prefix, config.connectivity_num_bins)

    # 5) Keep a clean output schema
    keep_cols = [
        config.poi_id_col,
        config.lat_col,
        config.lon_col,
        "region_coarse_token",
        "region_fine_token",
        "density_count",
        "density_bin",
        "nearest_graph_node_id",
        "node_degree",
        "connectivity_bin",
    ]

    descriptor_df = descriptor_df[keep_cols].copy()

    # Defensive uniqueness check
    if descriptor_df[config.poi_id_col].duplicated().any():
        dup_count = descriptor_df[config.poi_id_col].duplicated().sum()
        raise ValueError(f"POI ID column contains duplicates: {dup_count} duplicated rows found.")

    return descriptor_df