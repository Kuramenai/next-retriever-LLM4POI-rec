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

    def fit(self, case_base_df: pd.DataFrame) -> "DecisionStateEncoder":
        """
        Fit scalers on the training decision-state table.
        Must be called before transform/transform_single.
        """
        n = len(case_base_df)
        if n == 0:
            raise ValueError("Cannot fit encoder on empty case base.")

        # Collect raw arrays for each scalable block
        context_arr = np.zeros((n, 2), dtype=float)
        movement_dim = 4 * self.recent_k
        movement_arr = np.zeros((n, movement_dim), dtype=float)
        prefix_arr = np.zeros((n, 4), dtype=float)

        for i, (_, row) in enumerate(case_base_df.iterrows()):
            context_arr[i] = self._extract_context(row)
            movement_arr[i] = self._extract_movement(row)
            prefix_arr[i] = self._extract_prefix_summary(row)

        # Replace NaN with column mean for fitting
        for arr in [context_arr, movement_arr, prefix_arr]:
            col_means = np.nanmean(arr, axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            inds = np.where(np.isnan(arr))
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

        vectors = np.zeros((len(case_base_df), self._dim), dtype=float)
        for i, (_, row) in enumerate(case_base_df.iterrows()):
            vectors[i] = self._encode_row(row)
        return vectors

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
        coords = np.zeros((len(case_base_df), 2), dtype=float)
        for i, (_, row) in enumerate(case_base_df.iterrows()):
            coords[i, 0], coords[i, 1] = self._extract_coords(row)
        return coords

    @property
    def dim(self) -> int:
        return self._dim


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def retrieve_similar_decision_states(
    query_state: Union[pd.Series, pd.DataFrame],
    case_base_df: pd.DataFrame,
    encoder: DecisionStateEncoder,
    config,
    *,
    case_vectors: Optional[np.ndarray] = None,
    case_coords: Optional[np.ndarray] = None,
    top_k: int = 50,
    same_prototype_only: bool = True,
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

    if same_prototype_only and proto_col in q.index and proto_col in cands.columns:
        if pd.notna(q[proto_col]):
            proto_match = cands[proto_col] == q[proto_col]
            if proto_match.any():
                filter_mask &= proto_match

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
