"""
Decision-state retrieval via block-weighted continuous vector encoding
with explicit spatial distance kernel.

Architecture:
  - Non-spatial blocks (temporal, context, movement, prefix_summary, category)
    are encoded into a block-normalized vector for cosine similarity.
  - Spatial similarity is computed separately via an exponential decay kernel
    on haversine distance: exp(-distance / tau).
  - Final score is a weighted combination:
        score = alpha * spatial_score + (1 - alpha) * cosine_score
    where alpha = spatial_weight / total_weight.

This avoids the geometric error of L2-normalizing raw coordinates,
which collapses distance into angular direction from the city centroid.

Usage:
    encoder = DecisionStateEncoder(config)
    encoder.fit(case_base_df)
    case_vectors = encoder.transform(case_base_df)
    case_coords = encoder.extract_coords(case_base_df)

    result = retrieve_similar_decision_states(
        query_state, case_base_df, encoder, config,
        case_vectors=case_vectors,
        case_coords=case_coords,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from termcolor import cprint
from sklearn.preprocessing import StandardScaler
from spatial_encoding.extract_poi_spatial_descriptors import SpatialEncodingConfig


EARTH_RADIUS_M = 6_371_008.8


# ---------------------------------------------------------------------------
# Block weight configuration
# ---------------------------------------------------------------------------


@dataclass
class RetrievalBlockWeights:
    """
    Controls the relative importance of each feature block.

    `spatial` controls the weight of the haversine distance kernel.
    All other weights control the relative importance within the
    cosine-similarity vector, and the balance between cosine and
    spatial scores.

    Set to 0.0 to disable a block entirely.
    """

    spatial: float = 3.0  # haversine proximity (not in the vector)
    temporal: float = 1.5  # same time of day
    local_context: float = 1.0  # similar neighborhood density/connectivity
    movement: float = 2.0  # similar recent movement pattern
    prefix_summary: float = 1.0  # similar session stage
    category: float = 1.5  # same POI category at decision point

    @property
    def non_spatial_total(self) -> float:
        return (
            self.temporal
            + self.local_context
            + self.movement
            + self.prefix_summary
            + self.category
        )

    @property
    def total(self) -> float:
        return self.spatial + self.non_spatial_total

    @property
    def spatial_alpha(self) -> float:
        """Mixing coefficient for spatial vs. non-spatial scores."""
        t = self.total
        if t < 1e-12:
            return 0.0
        return self.spatial / t


# ---------------------------------------------------------------------------
# Spatial distance kernel
# ---------------------------------------------------------------------------


@dataclass
class SpatialKernelConfig:
    """
    Parameters for the exponential decay spatial kernel.

    spatial_score = exp(-haversine_meters / tau_m)

    tau_m controls the characteristic distance:
      - At distance = tau_m, score ≈ 0.37
      - At distance = 2*tau_m, score ≈ 0.14
      - At distance = 3*tau_m, score ≈ 0.05

    Default tau_m=500 means:
      - 100m away → score ≈ 0.82  (very similar)
      - 500m away → score ≈ 0.37  (moderate)
      - 1km away  → score ≈ 0.14  (weak)
      - 2km away  → score ≈ 0.02  (negligible)
    """

    tau_m: float = 500.0


def _haversine_one_to_many_m(
    lat1: float,
    lon1: float,
    lats2: np.ndarray,
    lons2: np.ndarray,
) -> np.ndarray:
    """Haversine distance in meters from one point to an array of points."""
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lats2_r = np.radians(lats2)
    lons2_r = np.radians(lons2)

    dlat = lats2_r - lat1_r
    dlon = lons2_r - lon1_r

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lats2_r) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


def _haversine_one_to_many_m_from_radians(
    lat1_r: float,
    lon1_r: float,
    lats2_r: np.ndarray,
    lons2_r: np.ndarray,
) -> np.ndarray:
    """
    Haversine distance in meters from one point to an array of points,
    with all inputs already in radians.
    """
    dlat = lats2_r - lat1_r
    dlon = lons2_r - lon1_r

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lats2_r) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


# ---------------------------------------------------------------------------
# Encoder (non-spatial blocks only)
# ---------------------------------------------------------------------------


class DecisionStateEncoder:
    """
    Encode decision-state rows into block-normalized continuous vectors
    for the non-spatial feature blocks, and extract coordinates separately
    for the spatial distance kernel.

    Non-spatial blocks
    ------------------
    temporal      : [sin(2π·h/24), cos(2π·h/24)]                    (2D)
    local_context : [density_count, node_degree]                     (2D)
    movement      : [log(d), log(g), sin(b), cos(b)] × recent_k     (4 × recent_k D)
    prefix_summary: [elapsed_min, repeat_ratio, unique_pois, unique_cats] (4D)
    category      : one-hot over known categories                    (|C| D)

    Spatial coordinates are NOT in the vector — they are extracted via
    extract_coords() and used by the retrieval function directly.
    """

    def __init__(
        self,
        config,
        weights: Optional[RetrievalBlockWeights] = None,
        spatial_kernel: Optional[SpatialKernelConfig] = None,
        recent_k: int = 2,
        *,
        lat_col: str = "curr_Latitude",
        lon_col: str = "curr_Longitude",
        density_col: str = "curr_density_count",
        connectivity_col: str = "curr_node_degree",
    ):
        self.config = config
        self.weights = weights or RetrievalBlockWeights()
        self.spatial_kernel = spatial_kernel or SpatialKernelConfig()
        self.recent_k = recent_k

        self.lat_col = lat_col
        self.lon_col = lon_col
        self.density_col = density_col
        self.connectivity_col = connectivity_col

        # Fitted state
        self._fitted = False
        self._context_scaler = StandardScaler()
        self._movement_scaler = StandardScaler()
        self._prefix_scaler = StandardScaler()
        self._category_vocab: list[str] = []
        self._category_to_idx: dict[str, int] = {}
        self._dim: int = 0

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    def _extract_coords(self, row) -> tuple[float, float]:
        """Extract (lat, lon) from a decision-state row."""
        lat = float(row.get(self.lat_col, np.nan))
        lon = float(row.get(self.lon_col, np.nan))
        return lat, lon

    def _extract_temporal(self, row) -> np.ndarray:
        ts = row.get("current_timestamp")
        if ts is not None and hasattr(ts, "hour"):
            hour = ts.hour + getattr(ts, "minute", 0) / 60.0
        else:
            hour = _time_bin_to_hour(row.get("current_time_bin", "midday"))
        angle = 2.0 * np.pi * hour / 24.0
        return np.array([np.sin(angle), np.cos(angle)], dtype=float)

    def _extract_context(self, row) -> np.ndarray:
        density = float(row.get(self.density_col, 0))
        connectivity = float(row.get(self.connectivity_col, 0))
        return np.array([density, connectivity], dtype=float)

    def _extract_movement(self, row) -> np.ndarray:
        parts = []
        for lag in range(1, self.recent_k + 1):
            prefix = f"prev{lag}"

            dist_m = row.get(f"{prefix}_distance_m", np.nan)
            gap_s = row.get(f"{prefix}_gap_s", np.nan)
            bearing = row.get(f"{prefix}_bearing_deg", np.nan)

            dist_m = float(dist_m) if pd.notna(dist_m) else np.nan
            gap_s = float(gap_s) if pd.notna(gap_s) else np.nan
            bearing_rad = np.radians(float(bearing)) if pd.notna(bearing) else np.nan

            parts.extend(
                [
                    np.log1p(dist_m) if not np.isnan(dist_m) else 0.0,
                    np.log1p(gap_s) if not np.isnan(gap_s) else 0.0,
                    np.sin(bearing_rad) if not np.isnan(bearing_rad) else 0.0,
                    np.cos(bearing_rad) if not np.isnan(bearing_rad) else 0.0,
                ]
            )
        return np.array(parts, dtype=float)

    def _extract_prefix_summary(self, row) -> np.ndarray:
        return np.array(
            [
                float(row.get("prefix_elapsed_min", 0)),
                float(row.get("prefix_repeat_ratio", 0)),
                float(row.get("prefix_unique_poi_count", 1)),
                float(row.get("prefix_unique_category_count", 1)),
            ],
            dtype=float,
        )

    def _extract_category_onehot(self, row) -> np.ndarray:
        vec = np.zeros(len(self._category_vocab), dtype=float)
        cat = row.get("current_category")
        if cat is not None and str(cat) in self._category_to_idx:
            vec[self._category_to_idx[str(cat)]] = 1.0
        return vec

    # ------------------------------------------------------------------
    # Fit: learn scalers and vocabulary from training case base
    # ------------------------------------------------------------------

    def _safe_numeric_col(
        self,
        df: pd.DataFrame,
        col: str,
        *,
        default: float = 0.0,
    ) -> np.ndarray:
        """
        Extract one numeric column as float32 array, coercing errors to NaN and
        filling NaN with `default`.
        """
        if col not in df.columns:
            return np.full(len(df), float(default), dtype=np.float32)
        arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        if np.isnan(arr).any():
            arr = np.where(np.isnan(arr), float(default), arr).astype(np.float32, copy=False)
        return arr

    def _l2_normalize_rows(self, mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return mat / norms

    def _extract_temporal_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized temporal block: [sin(2πh/24), cos(2πh/24)] per row.
        Uses current_timestamp if available, else falls back to time-bin midpoints.
        """
        if "current_timestamp" in df.columns:
            ts = pd.to_datetime(df["current_timestamp"], errors="coerce")
            hour = ts.dt.hour.astype(float) + ts.dt.minute.astype(float) / 60.0
            hour = hour.fillna(df.get("current_time_bin", "midday").map(_time_bin_to_hour))  # type: ignore[arg-type]
            hour = hour.to_numpy(dtype=np.float32)
        else:
            tbin = df.get("current_time_bin", pd.Series(["midday"] * len(df)))
            hour = tbin.map(_time_bin_to_hour).to_numpy(dtype=np.float32)  # type: ignore[arg-type]

        angle = 2.0 * np.pi * hour / 24.0
        return np.column_stack([np.sin(angle), np.cos(angle)]).astype(np.float32)

    def _extract_context_batch(self, df: pd.DataFrame) -> np.ndarray:
        density = self._safe_numeric_col(df, self.density_col, default=0.0)
        connectivity = self._safe_numeric_col(df, self.connectivity_col, default=0.0)
        return np.column_stack([density, connectivity]).astype(np.float32)

    def _extract_prefix_batch(self, df: pd.DataFrame) -> np.ndarray:
        a = self._safe_numeric_col(df, "prefix_elapsed_min", default=0.0)
        b = self._safe_numeric_col(df, "prefix_repeat_ratio", default=0.0)
        c = self._safe_numeric_col(df, "prefix_unique_poi_count", default=1.0)
        d = self._safe_numeric_col(df, "prefix_unique_category_count", default=1.0)
        return np.column_stack([a, b, c, d]).astype(np.float32)

    def _extract_movement_batch(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        for lag in range(1, self.recent_k + 1):
            prefix = f"prev{lag}"
            dist = self._safe_numeric_col(df, f"{prefix}_distance_m", default=0.0)
            gap = self._safe_numeric_col(df, f"{prefix}_gap_s", default=0.0)
            bearing = self._safe_numeric_col(df, f"{prefix}_bearing_deg", default=np.nan)

            dist = np.log1p(dist)
            gap = np.log1p(gap)

            bearing_rad = np.deg2rad(bearing.astype(np.float32, copy=False))
            sinb = np.sin(np.where(np.isnan(bearing_rad), 0.0, bearing_rad))
            cosb = np.cos(np.where(np.isnan(bearing_rad), 0.0, bearing_rad))

            parts.extend([dist, gap, sinb.astype(np.float32), cosb.astype(np.float32)])

        return np.column_stack(parts).astype(np.float32)

    def _extract_category_onehot_batch(self, df: pd.DataFrame) -> np.ndarray:
        if not self._category_vocab:
            return np.zeros((len(df), 0), dtype=np.float32)
        if "current_category" not in df.columns:
            return np.zeros((len(df), len(self._category_vocab)), dtype=np.float32)

        cats = df["current_category"].astype(str)
        codes = cats.map(self._category_to_idx).to_numpy(dtype=np.float32)
        # map returns floats with NaN for missing; convert to int with -1 for missing
        codes_i = np.where(np.isnan(codes), -1, codes).astype(np.int32)

        out = np.zeros((len(df), len(self._category_vocab)), dtype=np.float32)
        rows = np.arange(len(df), dtype=np.int32)
        valid = codes_i >= 0
        out[rows[valid], codes_i[valid]] = 1.0
        return out

    def fit(self, case_base_df: pd.DataFrame) -> "DecisionStateEncoder":
        """
        Fit scalers on the training decision-state table.
        Must be called before transform/transform_single.
        """
        n = len(case_base_df)
        if n == 0:
            raise ValueError("Cannot fit encoder on empty case base.")

        # Collect raw arrays for each scalable block (vectorized)
        context_arr = self._extract_context_batch(case_base_df)
        movement_arr = self._extract_movement_batch(case_base_df)
        prefix_arr = self._extract_prefix_batch(case_base_df)

        # Replace NaN with column mean for fitting
        for arr in [context_arr, movement_arr, prefix_arr]:
            col_means = np.nanmean(arr, axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            inds = np.where(np.isnan(arr))
            if inds[0].size > 0:
                arr[inds] = np.take(col_means, inds[1])

        self._context_scaler.fit(context_arr)
        self._movement_scaler.fit(movement_arr)
        self._prefix_scaler.fit(prefix_arr)

        # Category vocabulary (consistently stringified)
        if "current_category" in case_base_df.columns:
            cats = case_base_df["current_category"].dropna().unique().tolist()
            self._category_vocab = sorted(str(c) for c in cats)
        else:
            self._category_vocab = []
        self._category_to_idx = {c: i for i, c in enumerate(self._category_vocab)}

        # Total non-spatial vector dimension
        movement_dim = 4 * int(self.recent_k)
        self._dim = (
            2  # temporal
            + 2  # context
            + movement_dim  # movement
            + 4  # prefix summary
            + len(self._category_vocab)  # category
        )

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform: encode non-spatial blocks
    # ------------------------------------------------------------------

    def _encode_row(self, row) -> np.ndarray:
        """Encode a single row into a block-weighted non-spatial vector."""
        w = self.weights
        blocks = []

        # Temporal (already unit-scale by sin/cos construction)
        temporal = self._extract_temporal(row)
        blocks.append(_l2_normalize(temporal) * w.temporal)

        # Local context
        context = self._context_scaler.transform(
            self._extract_context(row).reshape(1, -1)
        )[0]
        blocks.append(_l2_normalize(context) * w.local_context)

        # Movement
        movement = self._movement_scaler.transform(
            self._extract_movement(row).reshape(1, -1)
        )[0]
        blocks.append(_l2_normalize(movement) * w.movement)

        # Prefix summary
        prefix = self._prefix_scaler.transform(
            self._extract_prefix_summary(row).reshape(1, -1)
        )[0]
        blocks.append(_l2_normalize(prefix) * w.prefix_summary)

        # Category
        cat_vec = self._extract_category_onehot(row)
        if len(cat_vec) > 0:
            blocks.append(_l2_normalize(cat_vec) * w.category)

        return np.concatenate(blocks)

    def transform_single(
        self,
        query_state: Union[pd.Series, pd.DataFrame, dict],
    ) -> np.ndarray:
        """Encode a single decision state into a 1-D non-spatial vector."""
        if not self._fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")

        if isinstance(query_state, pd.DataFrame):
            if len(query_state) != 1:
                raise ValueError("query_state DataFrame must contain exactly one row.")
            query_state = query_state.iloc[0]
        elif isinstance(query_state, dict):
            query_state = pd.Series(query_state)

        return self._encode_row(query_state)

    def transform(self, case_base_df: pd.DataFrame) -> np.ndarray:
        """Encode all rows into an (N × D) non-spatial matrix."""
        if not self._fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")
        df = case_base_df
        w = self.weights

        temporal = self._extract_temporal_batch(df)  # (N,2)
        temporal = self._l2_normalize_rows(temporal) * float(w.temporal)

        context = self._context_scaler.transform(self._extract_context_batch(df))
        context = self._l2_normalize_rows(context.astype(np.float32, copy=False)) * float(
            w.local_context
        )

        movement = self._movement_scaler.transform(self._extract_movement_batch(df))
        movement = self._l2_normalize_rows(movement.astype(np.float32, copy=False)) * float(
            w.movement
        )

        prefix = self._prefix_scaler.transform(self._extract_prefix_batch(df))
        prefix = self._l2_normalize_rows(prefix.astype(np.float32, copy=False)) * float(
            w.prefix_summary
        )

        cat = self._extract_category_onehot_batch(df)
        if cat.shape[1] > 0:
            cat = self._l2_normalize_rows(cat) * float(w.category)
            return np.concatenate([temporal, context, movement, prefix, cat], axis=1)

        return np.concatenate([temporal, context, movement, prefix], axis=1)

    # ------------------------------------------------------------------
    # Coordinate extraction (for spatial kernel)
    # ------------------------------------------------------------------

    def extract_coords_single(
        self,
        state: Union[pd.Series, pd.DataFrame, dict],
    ) -> tuple[float, float]:
        """Extract (lat, lon) from a single state."""
        if isinstance(state, pd.DataFrame):
            if len(state) != 1:
                raise ValueError("DataFrame must contain exactly one row.")
            state = state.iloc[0]
        elif isinstance(state, dict):
            state = pd.Series(state)
        return self._extract_coords(state)

    def extract_coords(self, case_base_df: pd.DataFrame) -> np.ndarray:
        """
        Extract (lat, lon) for all rows as an (N × 2) array.
        Pre-compute this alongside case_vectors for fast retrieval.
        """
        lat = self._safe_numeric_col(case_base_df, self.lat_col, default=np.nan)
        lon = self._safe_numeric_col(case_base_df, self.lon_col, default=np.nan)
        return np.column_stack([lat, lon]).astype(np.float32)

    @property
    def dim(self) -> int:
        return self._dim


# ---------------------------------------------------------------------------
# Retrieval index (precomputed, fast path)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionStateRetrievalIndex:
    """
    Precomputed retrieval structures for fast query-time similarity search.

    - `case_vectors_unit` should be float32 and row-L2-normalized.
    - `case_coords` should be float32 (N×2) lat/lon for spatial kernel.
    - `session_ids` and `prototype_ids` are used for cheap filtering.
    """

    case_base_df: pd.DataFrame
    case_vectors_unit: np.ndarray
    case_coords: Optional[np.ndarray]
    case_coords_rad: Optional[np.ndarray]
    session_ids: np.ndarray
    prototype_ids: np.ndarray
    all_idx: np.ndarray
    prototype_to_indices: dict[int, np.ndarray]


def build_retrieval_index(
    *,
    case_base_df: pd.DataFrame,
    case_vectors: np.ndarray,
    config,
    case_coords: Optional[np.ndarray] = None,
    proto_col: str = "proto_prototype_id",
) -> DecisionStateRetrievalIndex:
    """
    Build a retrieval index once at startup (numpy-first).
    """
    if len(case_base_df) == 0:
        raise ValueError("case_base_df is empty; cannot build retrieval index.")

    mat = np.asarray(case_vectors, dtype=np.float32)
    if mat.ndim != 2 or mat.shape[0] != len(case_base_df):
        raise ValueError(
            f"case_vectors must be shape (N,D) with N=len(case_base_df); got {mat.shape}."
        )

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    mat_unit = mat / norms

    if config.session_id_col not in case_base_df.columns:
        raise ValueError(
            f"case_base_df must contain {config.session_id_col!r} for filtering."
        )
    session_ids = case_base_df[config.session_id_col].to_numpy()

    if proto_col in case_base_df.columns:
        prototype_ids = pd.to_numeric(case_base_df[proto_col], errors="coerce").to_numpy()
    else:
        prototype_ids = np.full(len(case_base_df), np.nan, dtype=float)

    all_idx = np.arange(len(case_base_df), dtype=np.int64)

    prototype_to_indices: dict[int, np.ndarray] = {}
    valid = ~np.isnan(prototype_ids)
    if valid.any():
        for p in np.unique(prototype_ids[valid]).astype(int):
            prototype_to_indices[int(p)] = np.flatnonzero(prototype_ids == p)

    coords_arr: Optional[np.ndarray]
    coords_rad: Optional[np.ndarray]
    if case_coords is None:
        coords_arr = None
        coords_rad = None
    else:
        coords_arr = np.asarray(case_coords, dtype=np.float32)
        if coords_arr.shape != (len(case_base_df), 2):
            raise ValueError(f"case_coords must have shape (N,2); got {coords_arr.shape}.")
        # Precompute radians once to avoid per-query np.radians over large arrays
        coords_rad = np.radians(coords_arr.astype(np.float64)).astype(np.float32)

    return DecisionStateRetrievalIndex(
        case_base_df=case_base_df,
        case_vectors_unit=mat_unit,
        case_coords=coords_arr,
        case_coords_rad=coords_rad,
        session_ids=session_ids,
        prototype_ids=prototype_ids,
        all_idx=all_idx,
        prototype_to_indices=prototype_to_indices,
    )


# ---------------------------------------------------------------------------
# Retrieval (public API; can use index fast-path)
# ---------------------------------------------------------------------------

def retrieve_similar_decision_states(
    query_state: Union[pd.Series, pd.DataFrame],
    case_base_df: pd.DataFrame,
    encoder: DecisionStateEncoder,
    config,
    *,
    retrieval_index: Optional[DecisionStateRetrievalIndex] = None,
    case_vectors: Optional[np.ndarray] = None,
    case_coords: Optional[np.ndarray] = None,
    top_k: int = 50,
    same_prototype_only: bool = True,
    prototype_union_k: int = 1,
    exclude_same_session: bool = True,
) -> pd.DataFrame:
    """
    Retrieve top-k historical decision states most similar to the query.

    Scoring combines two components:
      1. Cosine similarity over non-spatial block vectors
         (temporal, context, movement, prefix, category)
      2. Spatial proximity via exp(-haversine / tau)

    Mixed as:
      score = alpha * spatial_score + (1 - alpha) * cosine_score
    where alpha = spatial_weight / total_weight.

    Parameters
    ----------
    query_state : Series or single-row DataFrame
        Output of build_current_decision_state().
    case_base_df : DataFrame
        Full training decision-state table.
    encoder : DecisionStateEncoder
        Fitted encoder.
    config : SpatialEncodingConfig
    case_vectors : ndarray, optional
        Pre-computed encoder.transform(case_base_df). Shape (N, D).
    case_coords : ndarray, optional
        Pre-computed encoder.extract_coords(case_base_df). Shape (N, 2).
    top_k : int
    same_prototype_only : bool
    prototype_union_k : int
        If `same_prototype_only=True`, restrict retrieval to the union of the top-M
        routed prototypes when the query contains `proto_top{i}_prototype_id`
        fields (e.g. from `Module1PrototypeRouter`). If those fields are absent,
        falls back to `proto_prototype_id`. Use 1 for hard top-1 routing (default),
        3 for a common accuracy/speed trade-off.
    exclude_same_session : bool

    Returns
    -------
    DataFrame: top-k rows from case_base_df with 'retrieval_score',
    'spatial_score', and 'cosine_score' columns.
    """
    # Normalize query
    if isinstance(query_state, pd.DataFrame):
        if len(query_state) != 1:
            raise ValueError("query_state DataFrame must contain exactly one row.")
        q = query_state.iloc[0]
    elif isinstance(query_state, pd.Series):
        q = query_state
    else:
        raise TypeError("query_state must be a pandas Series or single-row DataFrame.")

    # ------------------------------------------------------------------
    # Fast path: use pre-built retrieval index (avoid pandas until the end)
    # ------------------------------------------------------------------
    if retrieval_index is not None:
        idx = retrieval_index
        if len(idx.case_base_df) == 0:
            return idx.case_base_df.copy()

        cand_idx = idx.all_idx

        # Prototype bucket(s): hard top-1 or union of top-M router outputs
        if same_prototype_only:
            union_k = max(int(prototype_union_k), 1)
            proto_ids: list[int] = []

            # Prefer router top-M outputs when available (proto_top{i}_prototype_id)
            for i in range(1, union_k + 1):
                v = q.get(f"proto_top{i}_prototype_id", np.nan)
                if pd.notna(v):
                    try:
                        proto_ids.append(int(v))
                    except Exception:
                        pass

            # Fallback to single routed prototype id
            if not proto_ids:
                v = q.get("proto_prototype_id", np.nan)
                if pd.notna(v):
                    try:
                        proto_ids.append(int(v))
                    except Exception:
                        pass

            if proto_ids:
                buckets = [idx.prototype_to_indices.get(p) for p in dict.fromkeys(proto_ids)]
                buckets = [b for b in buckets if b is not None and b.size > 0]
                if buckets:
                    cand_idx = np.unique(np.concatenate(buckets))

        if exclude_same_session and config.session_id_col in q.index:
            qsid = q[config.session_id_col]
            cand_idx = cand_idx[idx.session_ids[cand_idx] != qsid]

        if cand_idx.size == 0:
            out = idx.case_base_df.iloc[[]].copy()
            out["retrieval_score"] = []
            out["spatial_score"] = []
            out["cosine_score"] = []
            return out

        # cosine over pre-normalized case vectors
        qvec = np.asarray(encoder.transform_single(q), dtype=np.float32)
        qnorm = np.linalg.norm(qvec)
        if qnorm > 1e-12:
            qvec = qvec / qnorm
        cosine_scores = idx.case_vectors_unit[cand_idx] @ qvec

        # spatial kernel (optional coords)
        query_lat, query_lon = encoder.extract_coords_single(q)
        if idx.case_coords is None or np.isnan(query_lat) or np.isnan(query_lon):
            spatial_scores = np.zeros(len(cand_idx), dtype=np.float32)
        else:
            # Prefer precomputed radians to avoid per-query np.radians over candidate arrays
            if idx.case_coords_rad is not None:
                coords_r = idx.case_coords_rad[cand_idx]
                lat1_r = float(np.radians(query_lat))
                lon1_r = float(np.radians(query_lon))
                distances_m = _haversine_one_to_many_m_from_radians(
                    lat1_r, lon1_r, coords_r[:, 0], coords_r[:, 1]
                )
            else:
                coords = idx.case_coords[cand_idx]
                distances_m = _haversine_one_to_many_m(
                    query_lat, query_lon, coords[:, 0], coords[:, 1]
                )
            tau = float(encoder.spatial_kernel.tau_m)
            spatial_scores = np.exp(-distances_m / tau).astype(np.float32)
            spatial_scores = np.where(np.isnan(spatial_scores), 0.0, spatial_scores)

        alpha = float(encoder.weights.spatial_alpha)
        combined_scores = alpha * spatial_scores + (1.0 - alpha) * cosine_scores

        if len(cand_idx) > top_k:
            local = np.argpartition(combined_scores, -top_k)[-top_k:]
            local = local[np.argsort(combined_scores[local])[::-1]]
        else:
            local = np.argsort(combined_scores)[::-1]

        rows = cand_idx[local]
        out = idx.case_base_df.iloc[rows].copy()
        out["retrieval_score"] = combined_scores[local]
        out["spatial_score"] = spatial_scores[local]
        out["cosine_score"] = cosine_scores[local]
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Reference path (older behavior)
    # ------------------------------------------------------------------
    cands = case_base_df.copy()
    if len(cands) == 0:
        return cands

    # ------------------------------------------------------------------
    # Hard filters
    # ------------------------------------------------------------------
    session_col = config.session_id_col
    proto_col = "proto_prototype_id"

    filter_mask = pd.Series(True, index=cands.index)

    if exclude_same_session and session_col in q.index and session_col in cands.columns:
        filter_mask &= cands[session_col] != q[session_col]

    if same_prototype_only and proto_col in cands.columns:
        union_k = max(int(prototype_union_k), 1)
        proto_vals: list[int] = []
        for i in range(1, union_k + 1):
            v = q.get(f"proto_top{i}_prototype_id", np.nan)
            if pd.notna(v):
                try:
                    proto_vals.append(int(v))
                except Exception:
                    pass
        if not proto_vals and proto_col in q.index and pd.notna(q[proto_col]):
            try:
                proto_vals.append(int(q[proto_col]))
            except Exception:
                proto_vals = []

        if proto_vals:
            proto_mask = cands[proto_col].isin(proto_vals)
            if proto_mask.any():
                filter_mask &= proto_mask

    filter_np = filter_mask.to_numpy()
    cands = cands.loc[filter_mask].copy()
    if len(cands) == 0:
        cands["retrieval_score"] = []
        cands["spatial_score"] = []
        cands["cosine_score"] = []
        return cands

    # ------------------------------------------------------------------
    # Non-spatial cosine similarity
    # ------------------------------------------------------------------
    query_vec = encoder.transform_single(q)

    if case_vectors is not None:
        filtered_vectors = case_vectors[filter_np]
    else:
        filtered_vectors = encoder.transform(cands)

    cosine_scores = _cosine_similarity_batch(query_vec, filtered_vectors)

    # ------------------------------------------------------------------
    # Spatial similarity via haversine kernel
    # ------------------------------------------------------------------
    query_lat, query_lon = encoder.extract_coords_single(q)

    if case_coords is not None:
        filtered_coords = case_coords[filter_np]
    else:
        filtered_coords = encoder.extract_coords(cands)

    if np.isnan(query_lat) or np.isnan(query_lon):
        # No spatial info available — fall back to cosine only
        spatial_scores = np.zeros(len(cands), dtype=float)
    else:
        distances_m = _haversine_one_to_many_m(
            query_lat,
            query_lon,
            filtered_coords[:, 0],
            filtered_coords[:, 1],
        )
        tau = encoder.spatial_kernel.tau_m
        spatial_scores = np.exp(-distances_m / tau)
        # NaN coords in case base → zero spatial score
        spatial_scores = np.where(np.isnan(spatial_scores), 0.0, spatial_scores)

    # ------------------------------------------------------------------
    # Combine scores
    # ------------------------------------------------------------------
    alpha = encoder.weights.spatial_alpha

    combined_scores = alpha * spatial_scores + (1.0 - alpha) * cosine_scores

    cands = cands.copy()
    cands["retrieval_score"] = combined_scores
    cands["spatial_score"] = spatial_scores
    cands["cosine_score"] = cosine_scores

    # ------------------------------------------------------------------
    # Select top-k
    # ------------------------------------------------------------------
    if len(cands) <= top_k:
        out = cands.sort_values("retrieval_score", ascending=False).reset_index(
            drop=True
        )
    else:
        top_indices = np.argpartition(combined_scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(combined_scores[top_indices])[::-1]]
        out = cands.iloc[top_indices].reset_index(drop=True)

    return out


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return np.zeros_like(vec)
    return vec / norm


def _cosine_similarity_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a query vector and each row of a matrix."""
    query_norm = np.linalg.norm(query)
    if query_norm < 1e-12:
        return np.zeros(len(matrix), dtype=float)

    row_norms = np.linalg.norm(matrix, axis=1)
    row_norms = np.where(row_norms < 1e-12, 1.0, row_norms)

    return (matrix @ query) / (row_norms * query_norm)


def _time_bin_to_hour(time_bin: str) -> float:
    """Approximate midpoint hour for a time bin label (fallback only)."""
    mapping = {
        "morning": 8.0,
        "midday": 13.0,
        "afternoon": 17.0,
        "evening": 21.0,
        "night": 2.0,
    }
    return mapping.get(str(time_bin), 12.0)


if __name__ == "__main__":
    city = "nyc"
    config = SpatialEncodingConfig()
    scrip_dir = Path(__file__).resolve().parent.parent
    encoder = DecisionStateEncoder(config=config)
    decision_state_table_df = pd.read_csv(
        scrip_dir / f"artifacts/{city}/{city}_decision_state_table.csv"
    )
    encoder.fit(decision_state_table_df)
    case_vectors = encoder.transform(decision_state_table_df)
    case_coords = encoder.extract_coords(decision_state_table_df)

    with open(scrip_dir / f"artifacts/{city}/{city}_case_vectors.pkl", "wb") as f:
        pickle.dump(case_vectors, f)

    with open(scrip_dir / f"artifacts/{city}/{city}_case_coords.pkl", "wb") as f:
        pickle.dump(case_coords, f)

    with open(scrip_dir / f"artifacts/{city}/{city}_case_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
