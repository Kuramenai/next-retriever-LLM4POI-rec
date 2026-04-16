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
from scipy.spatial import cKDTree
import h3

from pyrosm import OSM


import pickle


@dataclass
class SpatialEncodingConfig:
    h3_res_coarse: int = 8
    h3_res_fine: int = 9

    density_radius_m: float = 500.0
    density_num_bins: int = 5
    centrality_num_bins: int = 5

    use_urban_context: bool = False
    urban_context_radius_m: float = 150.0
    urban_context_priority_cols: Sequence[str] = (
        "amenity",
        "shop",
        "leisure",
        "tourism",
        "landuse",
        "highway",
    )

    # Column names in the incoming POI dataframe
    poi_id_col: str = "PId"
    lat_col: str = "Latitude"
    lon_col: str = "Longitude"

    # Output token prefixes
    density_prefix: str = "D"
    centrality_prefix: str = "C"

    topk_pair_neighbors: int = 200
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

    neighbor_indices = tree.query_ball_point(coords, r=radius_m, workers=-1)
    counts = np.array([max(len(idx_list) - 1, 0) for idx_list in neighbor_indices], dtype=int)

    return pd.Series(counts, index=poi_proj_gdf.index, name="density_count_500m")


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


def _compute_node_closeness_centrality(road_graph) -> Mapping[int, float]:
    """
    Heavy operation. Prefer caching the result outside this function if possible.
    Computes closeness centrality on the largest connected component
    of an undirected road graph, weighted by edge length.
    """
    warnings.warn(
        "Computing closeness centrality inside build_poi_spatial_descriptors(...). "
        "This can be slow on large city graphs; cache and pass node_centrality if possible.",
        RuntimeWarning,
    )

    try:
        G_u = ox.convert.to_undirected(road_graph)
    except Exception:
        G_u = road_graph.to_undirected()

    if not isinstance(G_u, nx.Graph):
        G_u = nx.Graph(G_u)

    if G_u.number_of_nodes() == 0:
        raise ValueError("Road graph is empty.")

    largest_cc_nodes = max(nx.connected_components(G_u), key=len)
    G_cc = G_u.subgraph(largest_cc_nodes).copy()

    centrality = nx.closeness_centrality(G_cc, distance="length")
    return centrality


def _normalize_context_token(raw_value: str, source_col: str) -> str:
    """
    Minimal normalization.
    Later, you can replace this with a coarser semantic collapse map.
    """
    token = str(raw_value).strip().lower().replace(" ", "_")
    token = token.replace("/", "_").replace("-", "_")
    return f"{source_col}:{token}"


def _resolve_urban_context_tokens(
    poi_proj_gdf: gpd.GeoDataFrame,
    context_gdf: Optional[gpd.GeoDataFrame],
    config: SpatialEncodingConfig,
) -> tuple[pd.Series, pd.Series]:
    """
    Optional fallback-based urban context assignment.
    For each POI, inspect nearby OSM context features within a radius and
    take the first available dominant signal from the configured priority columns.
    """
    unknown_token = pd.Series("unknown", index=poi_proj_gdf.index, dtype="object")
    unknown_source = pd.Series("none", index=poi_proj_gdf.index, dtype="object")

    # fmt:off

    if (
        (not config.use_urban_context)
        or (context_gdf is None)
        or (len(context_gdf) == 0)
    ):
        return unknown_token.rename("urban_context_token"), unknown_source.rename("urban_context_source")

    ctx = context_gdf.copy()
    if ctx.crs is None:
        raise ValueError("context_gdf must have a CRS.")
    ctx = ctx.to_crs(poi_proj_gdf.crs)

    # Keep only rows with valid geometries
    ctx = ctx[ctx.geometry.notna()].copy()
    if len(ctx) == 0:
        return unknown_token.rename("urban_context_token"), unknown_source.rename("urban_context_source")

    sindex = ctx.sindex

    tokens = []
    sources = []

    for geom in poi_proj_gdf.geometry:
        search_area = geom.buffer(config.urban_context_radius_m)
        candidate_idx = list(sindex.intersection(search_area.bounds))

        if not candidate_idx:
            tokens.append("unknown")
            sources.append("none")
            continue

        candidates = ctx.iloc[candidate_idx].copy()
        candidates = candidates[candidates.geometry.intersects(search_area)]

        chosen_token = "unknown"
        chosen_source = "none"

        if len(candidates) > 0:
            for col in config.urban_context_priority_cols:
                if col not in candidates.columns:
                    continue

                vals = candidates[col].dropna().astype(str).str.strip()
                vals = vals[vals.ne("")]

                if len(vals) == 0:
                    continue

                dominant_val = vals.mode().iloc[0]
                chosen_token = _normalize_context_token(dominant_val, col)
                chosen_source = col
                break

        tokens.append(chosen_token)
        sources.append(chosen_source)

    return (
        pd.Series(tokens, index=poi_proj_gdf.index, name="urban_context_token"),
        pd.Series(sources, index=poi_proj_gdf.index, name="urban_context_source"),
    )


def _get_undirected_graph(road_graph) -> nx.Graph:
    """
    Convert an OSMnx road graph to an undirected simple graph suitable
    for centrality computation.
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


def get_or_compute_node_closeness_centrality(
    road_graph,
    cache_path: str | Path,
    *,
    force_recompute: bool = False,
    largest_component_only: bool = True,
    distance_attr: str = "length",
    strict_cache_check: bool = True,
) -> Mapping[int, float]:
    """
    Load node closeness centrality from disk if available, otherwise compute and cache it.

    Parameters
    ----------
    road_graph:
        OSMnx road graph.
    cache_path:
        Path to a pickle cache file.
    force_recompute:
        If True, ignore any existing cache and recompute.
    largest_component_only:
        If True, compute centrality only on the graph's largest connected component.
        This is usually what you want for city road graphs.
    distance_attr:
        Edge attribute used as distance weight, usually 'length'.
    strict_cache_check:
        If True, raise an error when cache metadata does not match the current graph.
        If False, emit a warning and still use the cache.

    Returns
    -------
    centrality_dict:
        Mapping {node_id -> closeness_centrality}
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    current_sig = _graph_cache_signature(road_graph)

    if cache_path.exists() and not force_recompute:
        with cache_path.open("rb") as f:
            payload = pickle.load(f)

        if not isinstance(payload, dict) or "centrality" not in payload:
            raise ValueError(f"Cache file at {cache_path} has an unexpected format.")

        cached_sig = payload.get("graph_signature", {})
        cache_matches = cached_sig == current_sig

        if not cache_matches:
            msg = (
                f"Centrality cache signature mismatch for {cache_path}.\n"
                f"Cached:  {cached_sig}\n"
                f"Current: {current_sig}"
            )
            if strict_cache_check:
                raise ValueError(msg)
            warnings.warn(msg, RuntimeWarning)

        return payload["centrality"]

    # ---- Compute from scratch ----
    G_u = _get_undirected_graph(road_graph)

    if G_u.number_of_nodes() == 0:
        raise ValueError("Road graph is empty; cannot compute centrality.")

    if largest_component_only:
        largest_cc_nodes = max(nx.connected_components(G_u), key=len)
        G_work = G_u.subgraph(largest_cc_nodes).copy()
    else:
        G_work = G_u

    centrality = nx.closeness_centrality(G_work, distance=distance_attr)

    payload = {
        "metric": "closeness_centrality",
        "distance_attr": distance_attr,
        "largest_component_only": largest_component_only,
        "graph_signature": current_sig,
        "num_centrality_nodes": len(centrality),
        "centrality": centrality,
    }

    with cache_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    return centrality


def build_poi_spatial_descriptors(
    poi_df: pd.DataFrame,
    road_graph,
    config: SpatialEncodingConfig,
    node_centrality: Optional[Mapping[int, float]] = None,
    context_gdf: Optional[gpd.GeoDataFrame] = None,
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
    node_centrality:
        Optional precomputed mapping {node_id -> centrality_value}.
        If None, closeness centrality is computed internally.
    context_gdf:
        Optional GeoDataFrame for nearby OSM context features.
        Only used if config.use_urban_context=True.

    Returns
    -------
    descriptor_df:
        DataFrame keyed by POI ID with region, density, centrality,
        nearest node, and optional urban context descriptors.
    """
    _validate_poi_columns(poi_df, config)

    # 1) Base POI GeoDataFrames
    poi_gdf_wgs84 = _make_poi_gdf(poi_df, config)
    poi_proj_gdf = _project_gdf(poi_gdf_wgs84)

    # 2) Multi-scale H3 region tokens
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

    # 3) Density counts + bins
    density_counts = _compute_density_counts(
        poi_proj_gdf, radius_m=config.density_radius_m
    )
    descriptor_df["density_count_500m"] = density_counts.values
    descriptor_df["density_bin"] = _rank_bin(
        density_counts,
        prefix=config.density_prefix,
        num_bins=config.density_num_bins,
    ).values

    # 4) Nearest road-graph node
    nearest_nodes = _map_pois_to_nearest_graph_nodes(poi_gdf_wgs84, road_graph)
    descriptor_df["nearest_graph_node_id"] = nearest_nodes.values

    # 5) Centrality values + bins
    if node_centrality is None:
        node_centrality = _compute_node_closeness_centrality(road_graph)

    # fmt: off
    descriptor_df["road_centrality_value"] = descriptor_df["nearest_graph_node_id"].map(node_centrality)

    descriptor_df["centrality_bin"] = _rank_bin(
        descriptor_df["road_centrality_value"],
        prefix=config.centrality_prefix,
        num_bins=config.centrality_num_bins,
    ).values

    # 6) Optional urban context
    urban_context_token, urban_context_source = _resolve_urban_context_tokens(
        poi_proj_gdf=poi_proj_gdf,
        context_gdf=context_gdf,
        config=config,
    )
    descriptor_df["urban_context_token"] = urban_context_token.values
    descriptor_df["urban_context_source"] = urban_context_source.values

    # 7) Keep a clean output schema
    keep_cols = [
        config.poi_id_col,
        config.lat_col,
        config.lon_col,
        "region_coarse_token",
        "region_fine_token",
        "density_count_500m",
        "density_bin",
        "nearest_graph_node_id",
        "road_centrality_value",
        "centrality_bin",
        "urban_context_token",
        "urban_context_source",
    ]

    descriptor_df = descriptor_df[keep_cols].copy()

    # Defensive uniqueness check
    if descriptor_df[config.poi_id_col].duplicated().any():
        dup_count = descriptor_df[config.poi_id_col].duplicated().sum()
        raise ValueError(
            f"POI ID column contains duplicates: {dup_count} duplicated rows found."
        )

    return descriptor_df


city = "NYC"
out_dir = Path(f"preprocessed_data/{city}")
poi_df = pd.read_csv(out_dir / f"{city}.csv")
poi_df = poi_df.drop_duplicates()
poi_df = poi_df[["PId", "Latitude", "Longitude"]]

osm = OSM(f"geo_data/{city.lower()}.osm.pbf")
roads = osm.get_network(network_type="driving")
nodes, edges = osm.get_network(nodes=True, network_type="driving")
G_drive = osm.to_graph(nodes, edges, graph_type="networkx")

config = SpatialEncodingConfig(
    h3_res_coarse=8,
    h3_res_fine=9,
    density_radius_m=500.0,
    use_urban_context=False,  # keep off for now
)

# fmt:off
centrality_cache_path = f"artifacts/spatial/{city.lower()}_node_closeness_centrality.pkl"

node_centrality = get_or_compute_node_closeness_centrality(
    road_graph=G_drive,
    cache_path=centrality_cache_path,
    force_recompute=False,
    largest_component_only=True,
    distance_attr="length",
    strict_cache_check=True,
)


poi_spatial_df = build_poi_spatial_descriptors(
    poi_df=poi_df,  # must contain POIId, Latitude, Longitude
    road_graph=G_drive,  # OSMnx graph
    config=config,
    node_centrality=node_centrality,  # or pass a cached dict if already computed
    context_gdf=None,
)

poi_spatial_df.head()

# config.topk_pair_neighbors = 200
# config.road_distance_stretch_factor = 2.0
# config.distance_bin_edges_m = (250, 500, 1000, 2000, 5000)

# pair_transition_df = build_sparse_pair_transition_lookup(
#     poi_df=poi_spatial_df,   # output from build_poi_spatial_descriptors(...)
#     road_graph=G_drive,
#     config=config,
# )

# pair_transition_df.head()
