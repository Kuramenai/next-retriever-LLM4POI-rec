"""
Union candidate generation + learned reranker for next-POI prediction.

Pipeline:
    1. Generate candidates from both sources (spatial transitions + decision states)
    2. Extract per-candidate features
    3. Score with a trained model (logistic regression / LightGBM)
    4. Return top-k candidates with metadata for LLM stage

Usage
-----
    # Offline: build feature matrix from training data
    X_train, y_train, meta_train = build_reranker_training_data(
        decision_state_table_df=train_dst,
        transition_index=transition_index,
        retrieval_index=retrieval_index,
        encoder=encoder,
        config=config,
    )

    # Train the reranker
    reranker = train_reranker(X_train, y_train)

    # Online: evaluate
    metrics, details = evaluate_union_reranker(...)
"""

from __future__ import annotations

from collections import Counter
from typing import Optional, Union, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from termcolor import cprint
from tqdm import tqdm

import pickle
import time
from pathlib import Path
import concurrent.futures
import multiprocessing as mp

# Existing modules
from spatial_transition_retriever import (
    TransitionIndex,
    _generate_candidates_from_transitions,
    build_transition_index,
)
from retrieve_decisions_states import (
    DecisionStateEncoder,
    DecisionStateRetrievalIndex,
    retrieve_similar_decision_states,
)
from retrieve_candidates_pois import build_candidate_next_pois
from extract_poi_spatial_descriptors import SpatialEncodingConfig
from retrieve_decisions_states import build_retrieval_index
from session_decision_state_table import build_current_decision_state

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("lightgbm not installed. pip install lightgbm")


# ═══════════════════════════════════════════════════════════════════════════════
# Feature names (consistent ordering)
# ═══════════════════════════════════════════════════════════════════════════════
RERANKER_FEATURE_NAMES = [
    # Spatial transition features
    "st_transition_count",
    "st_transition_weight",
    "st_from_exact_poi",
    "st_n_source_pois",
    "st_best_source_weight",
    "st_context_score_max",
    "st_context_score_mean",
    "st_n_supporting_cases",
    # Decision-state retriever features
    "ds_in_candidates",
    "ds_rank",
    "ds_candidate_prob",
    "ds_max_case_score",
    "ds_mean_case_score",
    "ds_support_case_count",
    # Spatial features (raw — let model learn the right decay)
    "distance_to_candidate_m",
    "log_distance_to_candidate_m",
    # Category features
    "category_matches_current",
    "category_in_recent_prefix",
    # Source indicators
    "in_st_pool",
    "in_ds_pool",
    "in_both_pools",
]


def _as_query_series(query_state: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """Normalize a query-state input to the scalar Series used by the retrievers."""
    if isinstance(query_state, pd.DataFrame):
        if len(query_state) != 1:
            raise ValueError("query_state DataFrame must contain exactly one row.")
        return query_state.iloc[0]
    if isinstance(query_state, pd.Series):
        return query_state
    raise TypeError("query_state must be a pandas Series or single-row DataFrame.")


def _normalize_poi_id(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return int(value) if float(value).is_integer() else float(value)
    return str(value)


def split_checkins_by_session(
    checkins_df: pd.DataFrame,
    config,
    *,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split check-ins by whole sessions so prefixes from one session cannot leak
    across train/validation/test.
    """
    total = float(train_frac) + float(val_frac) + float(test_frac)
    if total <= 0:
        raise ValueError("Split fractions must sum to a positive value.")
    train_frac, val_frac, test_frac = train_frac / total, val_frac / total, test_frac / total

    session_ids = checkins_df[config.session_id_col].drop_duplicates().to_numpy()
    rng = np.random.default_rng(int(random_state))
    rng.shuffle(session_ids)

    n = len(session_ids)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train_ids = set(session_ids[:n_train])
    val_ids = set(session_ids[n_train : n_train + n_val])
    test_ids = set(session_ids[n_train + n_val :])

    train_df = checkins_df.loc[checkins_df[config.session_id_col].isin(train_ids)].copy()
    val_df = checkins_df.loc[checkins_df[config.session_id_col].isin(val_ids)].copy()
    test_df = checkins_df.loc[checkins_df[config.session_id_col].isin(test_ids)].copy()
    return train_df, val_df, test_df


# ═══════════════════════════════════════════════════════════════════════════════
# Union candidate generation
# ═══════════════════════════════════════════════════════════════════════════════


def _get_ds_candidates(
    query_state: pd.Series,
    retrieval_index: DecisionStateRetrievalIndex,
    encoder: DecisionStateEncoder,
    config,
    *,
    top_k_cases: int = 50,
    top_m_pois: int = 50,
    temperature: float = 0.2,
    exclude_same_session: bool = True,
) -> pd.DataFrame:
    """Run the decision-state retriever and return aggregated candidates."""
    retrieved_cases = retrieve_similar_decision_states(
        query_state=query_state,
        retrieval_index=retrieval_index,
        encoder=encoder,
        config=config,
        top_k=top_k_cases,
        same_prototype_only=False,
        exclude_same_session=exclude_same_session,
        prototype_union_k=3,
    )

    candidates = build_candidate_next_pois(
        retrieved_cases_df=retrieved_cases,
        config=config,
        top_m=top_m_pois,
        temperature=temperature,
    )

    return candidates


def _get_st_candidates_raw(
    current_poi_id: int,
    query_lat: float,
    query_lon: float,
    transition_index: TransitionIndex,
    *,
    nearby_radius_m: float = 1000.0,
    max_nearby_pois: int = 50,
    source_tau_m: float = 300.0,
    query_session_id=None,
    exclude_same_session: bool = True,
) -> dict[int, dict]:
    """Run spatial transition candidate generation (Stage 1 only, no reranking)."""
    return _generate_candidates_from_transitions(
        current_poi_id=current_poi_id,
        query_lat=query_lat,
        query_lon=query_lon,
        transition_index=transition_index,
        nearby_radius_m=nearby_radius_m,
        max_nearby_pois=max_nearby_pois,
        source_tau_m=source_tau_m,
        exclude_current=True,
        exclude_session_id=query_session_id if exclude_same_session else None,
    )


def _compute_context_scores(
    candidate_poi_ids: list[int],
    query_vec_unit: np.ndarray,
    transition_index: TransitionIndex,
    *,
    query_session_id=None,
    exclude_same_session: bool = True,
) -> dict[int, dict]:
    """
    Compute context similarity scores for a batch of candidate POIs.
    Returns {poi_id: {max, mean, n_cases}} for each candidate.
    """
    idx = transition_index
    candidate_cases: dict[int, np.ndarray] = {}
    all_case_indices = []

    for poi_id in candidate_poi_ids:
        case_indices = idx.next_poi_to_case_indices.get(poi_id)

        if case_indices is None or len(case_indices) == 0:
            candidate_cases[poi_id] = np.array([], dtype=np.int64)
            continue

        if exclude_same_session and query_session_id is not None:
            mask = idx.session_ids[case_indices] != query_session_id
            case_indices = case_indices[mask]

        if len(case_indices) == 0:
            candidate_cases[poi_id] = np.array([], dtype=np.int64)
            continue

        case_indices = np.asarray(case_indices, dtype=np.int64)
        candidate_cases[poi_id] = case_indices
        all_case_indices.append(case_indices)

    if not all_case_indices:
        return {poi_id: {"max": 0.0, "mean": 0.0, "n_cases": 0} for poi_id in candidate_poi_ids}

    unique_cases, inverse = np.unique(np.concatenate(all_case_indices), return_inverse=True)
    sims = idx.case_vectors_unit[unique_cases] @ query_vec_unit

    results = {}
    offset = 0
    for poi_id in candidate_poi_ids:
        case_indices = candidate_cases.get(poi_id)
        if case_indices is None or len(case_indices) == 0:
            results[poi_id] = {"max": 0.0, "mean": 0.0, "n_cases": 0}
            continue

        n_cases = len(case_indices)
        vals = sims[inverse[offset : offset + n_cases]]
        offset += n_cases
        results[poi_id] = {
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "n_cases": int(n_cases),
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Per-candidate feature extraction
# ═══════════════════════════════════════════════════════════════════════════════


def _get_next_category_map(transition_index: TransitionIndex) -> dict:
    """Cache next-POI category lookup on the transition index."""
    cached = getattr(transition_index, "_next_category_map", None)
    if cached is not None:
        return cached

    idx = transition_index
    if "next_category" not in idx.decision_state_df.columns:
        cached = {}
    else:
        cat_df = idx.decision_state_df.dropna(subset=["next_POIId"]).drop_duplicates(
            subset=["next_POIId"], keep="first"
        )
        cat_df = cat_df.assign(_poi_id_int=cat_df["next_POIId"].astype(int))
        cached = cat_df.set_index("_poi_id_int")["next_category"].to_dict()
    setattr(transition_index, "_next_category_map", cached)
    return cached


def _candidate_distances_m(
    candidate_poi_ids: list[int],
    query_lat: float,
    query_lon: float,
    next_poi_locations: dict[int, tuple[float, float]],
) -> np.ndarray:
    """Vectorized distance from query location to candidate POIs."""
    out = np.full(len(candidate_poi_ids), np.nan, dtype=np.float32)
    if np.isnan(query_lat) or np.isnan(query_lon):
        return out

    rows = []
    positions = []
    for pos, poi_id in enumerate(candidate_poi_ids):
        loc = next_poi_locations.get(poi_id)
        if loc is not None:
            positions.append(pos)
            rows.append(loc)

    if not rows:
        return out

    coords = np.asarray(rows, dtype=np.float64)
    lat1 = np.radians(float(query_lat))
    lon1 = np.radians(float(query_lon))
    lat2 = np.radians(coords[:, 0])
    lon2 = np.radians(coords[:, 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    distances = 6_371_008.8 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    out[np.asarray(positions, dtype=np.int32)] = distances.astype(np.float32)
    return out


def extract_candidate_features(
    candidate_poi_ids: list[int],
    query_state: Union[pd.Series, pd.DataFrame],
    *,
    st_candidates: dict[int, dict],
    ds_candidates_df: pd.DataFrame,
    context_scores: dict[int, dict],
    transition_index: TransitionIndex,
    config,
    recent_k: int = 4,
) -> pd.DataFrame:
    """
    Extract feature vectors for each candidate POI.

    Combines signals from both retrieval sources plus spatial and category features.

    Returns a DataFrame with one row per candidate, columns = RERANKER_FEATURE_NAMES
    plus metadata columns for the LLM stage.
    """
    query_state = _as_query_series(query_state)
    idx = transition_index
    candidate_poi_ids = [int(poi_id) for poi_id in candidate_poi_ids]

    # Current location
    query_lat = float(query_state.get("curr_Latitude", np.nan))
    query_lon = float(query_state.get("curr_Longitude", np.nan))
    raw_current_category = query_state.get("current_category", "")
    current_category = "" if pd.isna(raw_current_category) else str(raw_current_category)

    # Recent prefix categories
    recent_categories = set()
    for lag in range(1, recent_k + 1):
        cat_col = f"prev{lag}_category"
        if cat_col in query_state.index:
            cat = query_state.get(cat_col)
            if pd.notna(cat):
                recent_categories.add(str(cat))
    # Also check the current category
    if current_category:
        recent_categories.add(current_category)

    # DS candidate lookups, built without per-row pandas iteration.
    ds_rank: dict[int, int] = {}
    ds_candidate_prob: dict[int, float] = {}
    ds_max_case_score: dict[int, float] = {}
    ds_mean_case_score: dict[int, float] = {}
    ds_support_case_count: dict[int, int] = {}
    if ds_candidates_df is not None and len(ds_candidates_df) > 0:
        ds = ds_candidates_df.loc[ds_candidates_df["next_POIId"].notna()].copy()
        if len(ds) > 0:
            ds["_poi_id_int"] = ds["next_POIId"].astype(int)
            poi_arr = ds["_poi_id_int"].to_numpy()
            ds_rank = dict(zip(poi_arr, range(1, len(ds) + 1)))
            if "candidate_prob" in ds.columns:
                ds_candidate_prob = dict(zip(poi_arr, ds["candidate_prob"].to_numpy(dtype=float)))
            if "max_case_score" in ds.columns:
                ds_max_case_score = dict(zip(poi_arr, ds["max_case_score"].to_numpy(dtype=float)))
            if "mean_case_score" in ds.columns:
                ds_mean_case_score = dict(zip(poi_arr, ds["mean_case_score"].to_numpy(dtype=float)))
            if "support_case_count" in ds.columns:
                ds_support_case_count = dict(zip(poi_arr, ds["support_case_count"].to_numpy(dtype=int)))

    cat_map = _get_next_category_map(idx)
    cand_categories = [cat_map.get(poi_id, "") for poi_id in candidate_poi_ids]
    distances = _candidate_distances_m(candidate_poi_ids, query_lat, query_lon, idx.next_poi_locations)

    data = {
        "next_POIId": candidate_poi_ids,
        "next_category": cand_categories,
        "distance_m": distances,
        "st_transition_count": [
            st_candidates.get(poi_id, {}).get("transition_count", 0) for poi_id in candidate_poi_ids
        ],
        "st_transition_weight": [
            st_candidates.get(poi_id, {}).get("transition_weight", 0.0) for poi_id in candidate_poi_ids
        ],
        "st_from_exact_poi": [
            float(st_candidates.get(poi_id, {}).get("from_exact_poi", False)) for poi_id in candidate_poi_ids
        ],
        "st_n_source_pois": [
            len(st_candidates.get(poi_id, {}).get("source_pois", set())) for poi_id in candidate_poi_ids
        ],
        "st_best_source_weight": [
            st_candidates.get(poi_id, {}).get("best_source_weight", 0.0) for poi_id in candidate_poi_ids
        ],
        "st_context_score_max": [
            context_scores.get(poi_id, {}).get("max", 0.0) for poi_id in candidate_poi_ids
        ],
        "st_context_score_mean": [
            context_scores.get(poi_id, {}).get("mean", 0.0) for poi_id in candidate_poi_ids
        ],
        "st_n_supporting_cases": [
            context_scores.get(poi_id, {}).get("n_cases", 0) for poi_id in candidate_poi_ids
        ],
        "ds_in_candidates": [float(poi_id in ds_rank) for poi_id in candidate_poi_ids],
        "ds_rank": [float(ds_rank.get(poi_id, 999)) for poi_id in candidate_poi_ids],
        "ds_candidate_prob": [ds_candidate_prob.get(poi_id, 0.0) for poi_id in candidate_poi_ids],
        "ds_max_case_score": [ds_max_case_score.get(poi_id, 0.0) for poi_id in candidate_poi_ids],
        "ds_mean_case_score": [ds_mean_case_score.get(poi_id, 0.0) for poi_id in candidate_poi_ids],
        "ds_support_case_count": [ds_support_case_count.get(poi_id, 0) for poi_id in candidate_poi_ids],
        "distance_to_candidate_m": distances,
        "log_distance_to_candidate_m": np.log1p(distances),
        "category_matches_current": [
            float(bool(cat) and str(cat) == current_category) for cat in cand_categories
        ],
        "category_in_recent_prefix": [
            float(bool(cat) and str(cat) in recent_categories) for cat in cand_categories
        ],
        "in_st_pool": [float(poi_id in st_candidates) for poi_id in candidate_poi_ids],
        "in_ds_pool": [float(poi_id in ds_rank) for poi_id in candidate_poi_ids],
        "in_both_pools": [
            float((poi_id in st_candidates) and (poi_id in ds_rank)) for poi_id in candidate_poi_ids
        ],
    }

    return pd.DataFrame(data)


# ═══════════════════════════════════════════════════════════════════════════════
# Generate union candidates + features for one query
# ═══════════════════════════════════════════════════════════════════════════════


def generate_union_candidates_for_query(
    query_state: Union[pd.Series, pd.DataFrame],
    *,
    transition_index: TransitionIndex,
    retrieval_index: DecisionStateRetrievalIndex,
    encoder: DecisionStateEncoder,
    config,
    nearby_radius_m: float = 1000.0,
    source_tau_m: float = 300.0,
    ds_top_k_cases: int = 50,
    ds_top_m_pois: int = 50,
    ds_temperature: float = 0.2,
    recent_k: int = 4,
    exclude_same_session: bool = True,
) -> pd.DataFrame:
    """
    Run both retrievers, merge candidates, extract features.

    Returns a DataFrame where each row is a candidate POI with its full
    feature vector (for the learned reranker) and metadata (for LLM stage).
    """
    query_state = _as_query_series(query_state)
    current_poi = int(query_state.get("current_POIId"))
    query_lat = float(query_state.get("curr_Latitude", np.nan))
    query_lon = float(query_state.get("curr_Longitude", np.nan))
    query_session_id = query_state.get(config.session_id_col, None)

    # ── Source A: Spatial transition candidates ──────────────────────
    st_candidates = _get_st_candidates_raw(
        current_poi_id=current_poi,
        query_lat=query_lat,
        query_lon=query_lon,
        transition_index=transition_index,
        nearby_radius_m=nearby_radius_m,
        source_tau_m=source_tau_m,
        query_session_id=query_session_id,
        exclude_same_session=exclude_same_session,
    )

    # ── Source B: Decision-state retriever candidates ────────────────
    ds_candidates_df = _get_ds_candidates(
        query_state=query_state,
        retrieval_index=retrieval_index,
        encoder=encoder,
        config=config,
        top_k_cases=ds_top_k_cases,
        top_m_pois=ds_top_m_pois,
        temperature=ds_temperature,
        exclude_same_session=exclude_same_session,
    )

    # ── Union candidate set ──────────────────────────────────────────
    all_poi_ids = set(st_candidates.keys())
    if len(ds_candidates_df) > 0:
        all_poi_ids |= set(int(x) for x in ds_candidates_df["next_POIId"])
    all_poi_ids = sorted(all_poi_ids)

    if not all_poi_ids:
        return pd.DataFrame()

    # ── Context scores for all candidates ────────────────────────────
    query_vec = encoder.transform_single(query_state).astype(np.float32)
    qnorm = np.linalg.norm(query_vec)
    query_vec_unit = query_vec / qnorm if qnorm > 1e-12 else query_vec

    context_scores = _compute_context_scores(
        candidate_poi_ids=all_poi_ids,
        query_vec_unit=query_vec_unit,
        transition_index=transition_index,
        query_session_id=query_session_id,
        exclude_same_session=exclude_same_session,
    )

    # ── Feature extraction ───────────────────────────────────────────
    features_df = extract_candidate_features(
        candidate_poi_ids=all_poi_ids,
        query_state=query_state,
        st_candidates=st_candidates,
        ds_candidates_df=ds_candidates_df,
        context_scores=context_scores,
        transition_index=transition_index,
        config=config,
        recent_k=recent_k,
    )
    return features_df


# ═══════════════════════════════════════════════════════════════════════════════
# Training data generation
# ═══════════════════════════════════════════════════════════════════════════════


_RERANKER_WORKER_CONTEXT: dict[str, Any] = {}


def _init_reranker_worker(context: dict[str, Any]) -> None:
    """Install large read-only objects once per worker process."""
    global _RERANKER_WORKER_CONTEXT
    _RERANKER_WORKER_CONTEXT = context


def _process_reranker_query_chunk(query_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build reranker training rows for one chunk of query records."""
    ctx = _RERANKER_WORKER_CONTEXT
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_meta: list[pd.DataFrame] = []
    error_counts: Counter[str] = Counter()
    n_empty_pool = 0
    n_pool_hit = 0
    n_pool_miss = 0
    n_generated_queries = 0
    n_kept_queries = 0

    for rec in query_records:
        gold_poi = rec["gold_poi"]
        if pd.isna(gold_poi):
            continue
        gold_norm = _normalize_poi_id(gold_poi)

        try:
            if rec.get("prefix_df") is not None:
                query_state = build_current_decision_state(
                    partial_session_df=rec["prefix_df"],
                    poi_descriptor_df=ctx["poi_descriptor_df"],
                    config=ctx["config"],
                    lookup_df=ctx["lookup_df"],
                    coord_df=ctx["coord_df"],
                    recent_k=ctx["recent_k"],
                ).iloc[0]
            else:
                if rec.get("row_pos") is not None:
                    row = ctx["decision_state_table_df"].iloc[int(rec["row_pos"])]
                else:
                    row = rec["legacy_row"]
                query_state = row.drop(
                    labels=["next_POIId", "next_timestamp", "next_category"],
                    errors="ignore",
                )

            features_df = generate_union_candidates_for_query(
                query_state=query_state,
                transition_index=ctx["transition_index"],
                retrieval_index=ctx["retrieval_index"],
                encoder=ctx["encoder"],
                config=ctx["config"],
                nearby_radius_m=ctx["nearby_radius_m"],
                source_tau_m=ctx["source_tau_m"],
                ds_top_k_cases=ctx["ds_top_k_cases"],
                ds_top_m_pois=ctx["ds_top_m_pois"],
                ds_temperature=ctx["temperature"],
                recent_k=ctx["recent_k"],
                exclude_same_session=True,
            )

            if len(features_df) == 0:
                n_empty_pool += 1
                continue

            n_generated_queries += 1
            y_query = (
                features_df["next_POIId"]
                .apply(lambda x: _normalize_poi_id(x) == gold_norm)
                .to_numpy(dtype=np.float32)
            )
            pool_hit = bool(y_query.any())
            if pool_hit:
                n_pool_hit += 1
            else:
                n_pool_miss += 1
            if ctx["train_on_pool_hits_only"] and not pool_hit:
                continue

            X_query = features_df[RERANKER_FEATURE_NAMES].to_numpy(dtype=np.float32)
            X_query = np.nan_to_num(X_query, nan=0.0)

            meta_query = features_df[["next_POIId", "next_category", "distance_m"]].copy()
            meta_query["query_id"] = rec["query_id"]
            meta_query["gold_next_POIId"] = gold_poi
            meta_query["session_id"] = rec["session_id"]
            meta_query["decision_index"] = rec["decision_index"]
            meta_query["generated_pool_hit"] = pool_hit

            all_X.append(X_query)
            all_y.append(y_query)
            all_meta.append(meta_query)

        except Exception as e:
            error_counts[repr(e)] += 1

    return {
        "X": np.vstack(all_X) if all_X else None,
        "y": np.concatenate(all_y) if all_y else None,
        "meta": pd.concat(all_meta, ignore_index=True) if all_meta else None,
        "error_counts": dict(error_counts),
        "n_empty_pool": n_empty_pool,
        "n_pool_hit": n_pool_hit,
        "n_pool_miss": n_pool_miss,
        "n_generated_queries": n_generated_queries,
        "n_kept_queries": len(all_X),
    }


def _chunked(items: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    chunk_size = max(1, int(chunk_size))
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def build_reranker_training_data(
    decision_state_table_df: Optional[pd.DataFrame] = None,
    *,
    train_checkins_df: Optional[pd.DataFrame] = None,
    poi_descriptor_df: Optional[pd.DataFrame] = None,
    lookup_df: Optional[pd.DataFrame] = None,
    coord_df: Optional[pd.DataFrame] = None,
    transition_index: TransitionIndex,
    retrieval_index: DecisionStateRetrievalIndex,
    encoder: DecisionStateEncoder,
    config,
    nearby_radius_m: float = 1000.0,
    source_tau_m: float = 300.0,
    ds_top_k_cases: int = 50,
    ds_top_m_pois: int = 50,
    recent_k: Optional[int] = None,
    max_samples: Optional[int] = None,
    max_queries_per_session: Optional[int] = 1,
    query_state_source: str = "decision_state",
    train_on_pool_hits_only: bool = True,
    temperature: float = 0.2,
    exclude_same_session: bool = True,
    random_state: int = 42,
    show_progress: bool = True,
    max_workers: int = 1,
    chunk_size: int = 500,
    mp_start_method: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Generate (X, y, meta) training data for the learned reranker.

    For full training, ``query_state_source="decision_state"`` reuses the
    precomputed decision-state table and drops label columns before retrieval.
    Use ``query_state_source="prefix"`` to reconstruct each query from raw
    check-in prefixes via ``build_current_decision_state`` for parity checks.
    """

    cprint("Building reranker training data...", "yellow")

    encoder_recent_k = int(getattr(encoder, "recent_k", 2))
    if recent_k is None:
        recent_k = encoder_recent_k
    else:
        recent_k = int(recent_k)
        if recent_k != encoder_recent_k:
            raise ValueError(f"recent_k={recent_k} does not match encoder.recent_k={encoder_recent_k}.")

    rng = np.random.default_rng(int(random_state))
    query_records: list[dict[str, Any]] = []
    if query_state_source not in {"decision_state", "prefix"}:
        raise ValueError("query_state_source must be either 'decision_state' or 'prefix'.")

    if query_state_source == "decision_state":
        if decision_state_table_df is None:
            raise ValueError("query_state_source='decision_state' requires decision_state_table_df.")
        cprint("Using precomputed decision states as reranker query states.", "yellow")
        dst = decision_state_table_df.copy().reset_index(drop=True)
        if config.session_id_col not in dst.columns:
            raise ValueError(f"decision_state_table_df missing {config.session_id_col!r}.")

        for session_id, session_dst in dst.groupby(config.session_id_col, sort=False):
            if len(session_dst) == 0:
                continue
            row_positions = np.arange(len(session_dst), dtype=int)
            if max_queries_per_session is not None and len(row_positions) > max_queries_per_session:
                row_positions = np.sort(
                    rng.choice(row_positions, size=int(max_queries_per_session), replace=False)
                )
            for local_pos in row_positions:
                row = session_dst.iloc[int(local_pos)]
                gold_poi = row.get("next_POIId")
                if pd.isna(gold_poi):
                    continue
                decision_index = row.get("decision_index", local_pos)
                row_pos = int(row.name)
                query_records.append(
                    {
                        "query_id": f"{session_id}:{int(decision_index)}",
                        "session_id": session_id,
                        "decision_index": decision_index,
                        "prefix_df": None,
                        "legacy_row": None,
                        "row_pos": row_pos,
                        "gold_poi": gold_poi,
                    }
                )

    elif train_checkins_df is not None:
        required_inputs = {
            "poi_descriptor_df": poi_descriptor_df,
            "lookup_df": lookup_df,
            "coord_df": coord_df,
        }
        missing_inputs = [name for name, value in required_inputs.items() if value is None]
        if missing_inputs:
            raise ValueError(f"train_checkins_df mode requires these inputs: {missing_inputs}")

        required_cols = [config.session_id_col, config.timestamp_col, config.poi_id_col]
        missing_cols = [c for c in required_cols if c not in train_checkins_df.columns]
        if missing_cols:
            raise ValueError(f"train_checkins_df missing required columns: {missing_cols}")

        work = train_checkins_df.copy()
        work[config.timestamp_col] = pd.to_datetime(work[config.timestamp_col], errors="coerce")
        work = work.loc[work[config.timestamp_col].notna()].copy()
        work = work.sort_values(required_cols).reset_index(drop=True)

        sort_session_cols = [config.timestamp_col, config.poi_id_col]
        for session_id, session_df in work.groupby(config.session_id_col, sort=False):
            session_df = session_df.sort_values(sort_session_cols).reset_index(drop=True)
            if len(session_df) < 2:
                continue

            decision_indices = np.arange(len(session_df) - 1, dtype=int)
            if max_queries_per_session is not None and len(decision_indices) > max_queries_per_session:
                decision_indices = np.sort(
                    rng.choice(decision_indices, size=int(max_queries_per_session), replace=False)
                )

            for decision_index in decision_indices:
                prefix_df = session_df.iloc[: decision_index + 1].copy().reset_index(drop=True)
                gold_poi = session_df.iloc[decision_index + 1][config.poi_id_col]
                if len(prefix_df) < 2:
                    continue
                query_records.append(
                    {
                        "query_id": f"{session_id}:{int(decision_index)}",
                        "session_id": session_id,
                        "decision_index": int(decision_index),
                        "prefix_df": prefix_df,
                        "legacy_row": None,
                        "row_pos": None,
                        "gold_poi": gold_poi,
                    }
                )
    else:
        raise ValueError("query_state_source='prefix' requires train_checkins_df.")

    if max_samples is not None and len(query_records) > int(max_samples):
        selected = rng.choice(len(query_records), size=int(max_samples), replace=False)
        query_records = [query_records[int(i)] for i in selected]

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_meta: list[pd.DataFrame] = []
    error_counts: Counter[str] = Counter()
    n_empty_pool = 0
    n_pool_hit = 0
    n_pool_miss = 0
    n_generated_queries = 0
    n_kept_queries = 0

    worker_context = {
        "decision_state_table_df": decision_state_table_df,
        "poi_descriptor_df": poi_descriptor_df,
        "lookup_df": lookup_df,
        "coord_df": coord_df,
        "transition_index": transition_index,
        "retrieval_index": retrieval_index,
        "encoder": encoder,
        "config": config,
        "nearby_radius_m": nearby_radius_m,
        "source_tau_m": source_tau_m,
        "temperature": temperature,
        "ds_top_k_cases": ds_top_k_cases,
        "ds_top_m_pois": ds_top_m_pois,
        "recent_k": recent_k,
        "train_on_pool_hits_only": train_on_pool_hits_only,
        "exclude_same_session": exclude_same_session,
    }

    chunks = _chunked(query_records, chunk_size)

    def _merge_chunk_result(result: dict[str, Any]) -> None:
        nonlocal n_empty_pool, n_pool_hit, n_pool_miss, n_generated_queries, n_kept_queries
        if result["X"] is not None:
            all_X.append(result["X"])
            all_y.append(result["y"])
            all_meta.append(result["meta"])
        error_counts.update(result["error_counts"])
        n_empty_pool += int(result["n_empty_pool"])
        n_pool_hit += int(result["n_pool_hit"])
        n_pool_miss += int(result["n_pool_miss"])
        n_generated_queries += int(result["n_generated_queries"])
        n_kept_queries += int(result["n_kept_queries"])

    max_workers = max(1, int(max_workers))
    if max_workers == 1 or len(chunks) <= 1:
        _init_reranker_worker(worker_context)
        iterator = (
            tqdm(chunks, total=len(chunks), desc="Building reranker training data", unit="chunk")
            if show_progress
            else chunks
        )
        for chunk in iterator:
            _merge_chunk_result(_process_reranker_query_chunk(chunk))
    else:
        cprint(
            f"Building reranker training data with {max_workers} workers and chunk_size={int(chunk_size)}...",
            "yellow",
        )
        mp_context = mp.get_context(mp_start_method) if mp_start_method else None
        executor_kwargs = {
            "max_workers": max_workers,
            "initializer": _init_reranker_worker,
            "initargs": (worker_context,),
        }
        if mp_context is not None:
            executor_kwargs["mp_context"] = mp_context
        with concurrent.futures.ProcessPoolExecutor(**executor_kwargs) as executor:
            future_to_size = {
                executor.submit(_process_reranker_query_chunk, chunk): len(chunk) for chunk in chunks
            }
            pbar = (
                tqdm(total=len(query_records), desc="Building reranker training data", unit="query")
                if show_progress
                else None
            )
            try:
                for future in concurrent.futures.as_completed(future_to_size):
                    _merge_chunk_result(future.result())
                    if pbar is not None:
                        pbar.update(future_to_size[future])
            finally:
                if pbar is not None:
                    pbar.close()

    if error_counts:
        cprint("Skipped reranker training queries because of errors:", "yellow")
        for err, count in error_counts.most_common(5):
            cprint(f"  {count:>5}  {err}", "yellow")
    if n_empty_pool:
        cprint(f"Skipped {n_empty_pool} queries with an empty union candidate pool.", "yellow")
    if n_pool_miss:
        cprint(
            f"Skipped {n_pool_miss} queries where the gold POI was absent from the union pool.",
            "yellow",
        )
    training_pool_recall = n_pool_hit / n_generated_queries if n_generated_queries else 0.0
    cprint(
        "Reranker query diagnostics: "
        f"total={len(query_records)}, generated_pool={n_generated_queries}, "
        f"pool_hit={n_pool_hit}, pool_miss={n_pool_miss}, "
        f"empty_pool={n_empty_pool}, errors={sum(error_counts.values())}, "
        f"training_pool_recall={training_pool_recall:.4f}",
        "cyan",
    )

    if not all_X:
        raise RuntimeError("No training data generated. Check inputs and candidate pool recall.")

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    meta = pd.concat(all_meta, ignore_index=True)

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    cprint(
        f"Reranker training data: {len(X)} candidates "
        f"({n_pos} positive, {n_neg} negative, "
        f"ratio 1:{n_neg // max(n_pos, 1)}), "
        f"from {n_kept_queries} queries.",
        "green",
    )

    return X, y, meta


# ═══════════════════════════════════════════════════════════════════════════════
# Reranker training
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrainedReranker:
    """Wrapper for a trained reranker model + fitted scaler."""

    model: Any
    scaler: StandardScaler
    feature_names: list[str]
    model_type: str


def train_reranker(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_type: str = "logistic",
    random_state: int = 42,
) -> TrainedReranker:
    """
    Train a learned reranker.

    Parameters
    ----------
    model_type:
        "logistic" — LogisticRegression (fast, interpretable baseline)
        "lgbm" — LightGBM (better performance, needs lightgbm installed)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.nan_to_num(X, nan=0.0))
    if np.unique(y).size < 2:
        raise ValueError("Reranker training labels must contain both positive and negative examples.")

    if model_type == "logistic":
        # Class imbalance: gold POI is ~1 out of ~250 candidates
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            random_state=random_state,
        )
        model.fit(X_scaled, y)

        # Print learned weights for interpretability
        cprint("\nLogistic regression coefficients:", "cyan")
        for name, coef in sorted(
            zip(RERANKER_FEATURE_NAMES, model.coef_[0]),
            key=lambda x: abs(x[1]),
            reverse=True,
        ):
            cprint(f"  {name:<35s} {coef:>8.4f}", "cyan")

    elif model_type == "lgbm":
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)

        model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=n_neg / max(n_pos, 1),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbose=-1,
        )
        model.fit(X_scaled, y)

        cprint("\nLightGBM feature importances:", "cyan")
        for name, imp in sorted(
            zip(RERANKER_FEATURE_NAMES, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        ):
            cprint(f"  {name:<35s} {imp:>8d}", "cyan")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return TrainedReranker(
        model=model,
        scaler=scaler,
        feature_names=list(RERANKER_FEATURE_NAMES),
        model_type=model_type,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Inference: score + rank with trained reranker
# ═══════════════════════════════════════════════════════════════════════════════


def rerank_candidates(
    features_df: pd.DataFrame,
    reranker: TrainedReranker,
    *,
    top_m: int = 20,
) -> pd.DataFrame:
    """
    Score and rank candidates using the trained reranker.

    Returns top-m candidates sorted by predicted probability.
    """
    if len(features_df) == 0:
        return features_df

    X = features_df[reranker.feature_names].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0)
    X_scaled = reranker.scaler.transform(X)

    proba = reranker.model.predict_proba(X_scaled)[:, 1]

    result = features_df.copy()
    result["reranker_score"] = proba
    result = result.sort_values("reranker_score", ascending=False).reset_index(drop=True)

    return result.head(top_m)


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end: union generate → rerank for one query
# ═══════════════════════════════════════════════════════════════════════════════


def retrieve_and_rerank(
    query_state: pd.Series,
    *,
    transition_index: TransitionIndex,
    retrieval_index: DecisionStateRetrievalIndex,
    encoder: DecisionStateEncoder,
    reranker: TrainedReranker,
    config,
    top_m: int = 20,
    nearby_radius_m: float = 1000.0,
    source_tau_m: float = 300.0,
    ds_top_k_cases: int = 50,
    ds_top_m_pois: int = 50,
    recent_k: int = 4,
) -> dict:
    """
    Full pipeline: union candidates → feature extraction → learned reranking.
    """
    features_df = generate_union_candidates_for_query(
        query_state=query_state,
        transition_index=transition_index,
        retrieval_index=retrieval_index,
        encoder=encoder,
        config=config,
        nearby_radius_m=nearby_radius_m,
        source_tau_m=source_tau_m,
        ds_top_k_cases=ds_top_k_cases,
        ds_top_m_pois=ds_top_m_pois,
        recent_k=recent_k,
    )

    if len(features_df) == 0:
        return {
            "candidate_pois": pd.DataFrame(),
            "n_candidates_generated": 0,
            "generated_candidate_ids": [],
        }

    ranked = rerank_candidates(features_df, reranker, top_m=top_m)

    return {
        "candidate_pois": ranked,
        "n_candidates_generated": len(features_df),
        "generated_candidate_ids": features_df["next_POIId"].tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_union_reranker(
    test_checkins_df: pd.DataFrame,
    *,
    poi_descriptor_df: pd.DataFrame,
    lookup_df: pd.DataFrame,
    coord_df: pd.DataFrame,
    transition_index: TransitionIndex,
    retrieval_index: DecisionStateRetrievalIndex,
    encoder: DecisionStateEncoder,
    reranker: TrainedReranker,
    config,
    k_values: tuple[int, ...] = (1, 3, 5, 10, 20),
    top_m_pois: int = 20,
    nearby_radius_m: float = 1000.0,
    source_tau_m: float = 300.0,
    ds_top_k_cases: int = 50,
    ds_top_m_pois: int = 50,
    recent_k: Optional[int] = None,
    min_checkins: int = 2,
    max_sessions: Optional[int] = None,
    random_state: int = 42,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the union + learned reranker pipeline."""

    def _normalize(v):
        if pd.isna(v):
            return None
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            return int(v) if float(v).is_integer() else float(v)
        return str(v)

    k_values = tuple(sorted({int(k) for k in k_values if int(k) > 0}))
    top_m_pois = max(top_m_pois, max(k_values))

    encoder_recent_k = int(getattr(encoder, "recent_k", 2))
    if recent_k is None:
        recent_k = encoder_recent_k
    else:
        recent_k = int(recent_k)
        if recent_k != encoder_recent_k:
            raise ValueError(f"recent_k={recent_k} does not match encoder.recent_k={encoder_recent_k}.")

    work = test_checkins_df.copy()
    work[config.timestamp_col] = pd.to_datetime(work[config.timestamp_col], errors="coerce")
    work = work.loc[work[config.timestamp_col].notna()].copy()
    work = work.sort_values([config.session_id_col, config.timestamp_col, config.poi_id_col]).reset_index(
        drop=True
    )

    session_ids = work[config.session_id_col].drop_duplicates().tolist()
    if max_sessions is not None and len(session_ids) > max_sessions:
        rng = np.random.default_rng(random_state)
        session_ids = rng.choice(session_ids, size=max_sessions, replace=False).tolist()
        work = work.loc[work[config.session_id_col].isin(set(session_ids))].copy()

    groups = work.groupby(config.session_id_col, sort=False)
    iterator = tqdm(groups, desc="evaluate union reranker", unit="session") if show_progress else groups

    rows = []
    for session_id, session_df in iterator:
        session_df = session_df.sort_values([config.timestamp_col, config.poi_id_col]).reset_index(drop=True)

        base_row = {
            config.session_id_col: session_id,
            "n_checkins": len(session_df),
            "skipped": False,
            "error": None,
        }

        if len(session_df) < min_checkins:
            rows.append({**base_row, "skipped": True, "gold_next_POIId": None, "gold_rank": None})
            continue

        prefix_df = session_df.iloc[:-1].copy().reset_index(drop=True)
        gold_poi_id = session_df.iloc[-1][config.poi_id_col]

        try:
            query_state = build_current_decision_state(
                partial_session_df=prefix_df,
                poi_descriptor_df=poi_descriptor_df,
                config=config,
                lookup_df=lookup_df,
                coord_df=coord_df,
                recent_k=recent_k,
            )

            result = retrieve_and_rerank(
                query_state=query_state,
                transition_index=transition_index,
                retrieval_index=retrieval_index,
                encoder=encoder,
                reranker=reranker,
                config=config,
                top_m=top_m_pois,
                nearby_radius_m=nearby_radius_m,
                source_tau_m=source_tau_m,
                ds_top_k_cases=ds_top_k_cases,
                ds_top_m_pois=ds_top_m_pois,
                recent_k=recent_k,
            )

            candidate_pois = result["candidate_pois"]
            candidate_ids = candidate_pois["next_POIId"].tolist() if len(candidate_pois) > 0 else []
            generated_ids = result.get("generated_candidate_ids", [])

            # Rank of gold
            rank = None
            gold_norm = _normalize(gold_poi_id)
            for i, cid in enumerate(candidate_ids, start=1):
                if _normalize(cid) == gold_norm:
                    rank = i
                    break

            pool_hit = gold_norm in {_normalize(x) for x in generated_ids}

            rec = {
                **base_row,
                "gold_next_POIId": gold_poi_id,
                "gold_rank": rank,
                "candidate_count": len(candidate_pois),
                "n_candidates_generated": result["n_candidates_generated"],
                "generated_pool_hit": pool_hit,
                "generated_pool_size": len(generated_ids),
            }
            for k in k_values:
                rec[f"hit@{k}"] = bool(rank is not None and rank <= k)
                rec[f"recall@{k}"] = float(rec[f"hit@{k}"])
            rows.append(rec)

        except Exception as e:
            rec = {**base_row, "error": repr(e), "gold_next_POIId": gold_poi_id, "gold_rank": None}
            for k in k_values:
                rec[f"hit@{k}"] = False
                rec[f"recall@{k}"] = 0.0
            rows.append(rec)

    details_df = pd.DataFrame(rows)
    valid_mask = (~details_df["skipped"].fillna(False)) & details_df["error"].isna()
    valid = details_df.loc[valid_mask]

    summary = {
        "n_sessions_total": len(details_df),
        "n_sessions_evaluated": len(valid),
        "n_sessions_error": int(((~details_df["skipped"].fillna(False)) & details_df["error"].notna()).sum()),
    }
    if len(valid) > 0:
        ranks = pd.to_numeric(valid["gold_rank"], errors="coerce")
        for k in k_values:
            summary[f"hit@{k}"] = float(valid[f"hit@{k}"].mean())
            summary[f"recall@{k}"] = summary[f"hit@{k}"]
        summary["mrr"] = float((1.0 / ranks.dropna()).sum() / len(valid))
        summary["mean_gold_rank_when_hit"] = float(ranks.dropna().mean()) if ranks.notna().any() else np.nan
        summary["mean_gold_rank"] = summary["mean_gold_rank_when_hit"]
        summary["generated_pool_recall"] = float(valid["generated_pool_hit"].astype(float).mean())
        summary["mean_generated_pool_size"] = float(valid["generated_pool_size"].mean())
        summary["median_generated_pool_size"] = float(valid["generated_pool_size"].median())
        summary["mean_returned_candidate_count"] = float(valid["candidate_count"].mean())
        summary["reranking_loss@20"] = summary["generated_pool_recall"] - summary.get("recall@20", 0)
    else:
        for k in k_values:
            summary[f"hit@{k}"] = np.nan
            summary[f"recall@{k}"] = np.nan
        summary["mrr"] = np.nan
        summary["mean_gold_rank_when_hit"] = np.nan
        summary["mean_gold_rank"] = np.nan
        summary["generated_pool_recall"] = np.nan
        summary["mean_generated_pool_size"] = np.nan
        summary["median_generated_pool_size"] = np.nan
        summary["mean_returned_candidate_count"] = np.nan
        summary["reranking_loss@20"] = np.nan
        error_counts = details_df.loc[
            (~details_df["skipped"].fillna(False)) & details_df["error"].notna(),
            "error",
        ].value_counts()
        summary["top_error"] = error_counts.index[0] if not error_counts.empty else None

    return pd.DataFrame([summary]), details_df


if __name__ == "__main__":
    city = "nyc"
    k_values = (1, 3, 5, 10, 20)

    scrip_dir = Path(__file__).resolve().parent.parent
    decision_state_table_df = pd.read_csv(scrip_dir / f"artifacts/{city}/{city}_decision_state_table.csv")

    poi_descriptor_df = pd.read_csv(scrip_dir / f"artifacts/{city}/{city}_poi_descriptor.csv")

    with open(scrip_dir / f"artifacts/{city}/{city}_pair_lookup.pkl", "rb") as f:
        pair_lookup = pickle.load(f)
    with open(scrip_dir / f"artifacts/{city}/{city}_poi_coord_map.pkl", "rb") as f:
        poi_coord_map = pickle.load(f)
    lookup_df = pd.DataFrame.from_dict(pair_lookup, orient="index")
    lookup_df.index = pd.MultiIndex.from_tuples(lookup_df.index, names=["src_POIId", "dst_POIId"])  # fmt: skip
    coord_df = pd.DataFrame.from_dict(poi_coord_map, orient="index")

    config = SpatialEncodingConfig()

    recent_k = 4
    encoder = DecisionStateEncoder(config=config, recent_k=recent_k)
    encoder.fit(decision_state_table_df)
    case_vectors = encoder.transform(decision_state_table_df)
    case_coords = encoder.extract_coords(decision_state_table_df)

    sid_col = config.session_id_col
    ts_col = config.timestamp_col
    poi_id_col = config.poi_id_col

    test_checkins = pd.read_csv(scrip_dir / f"data/{city}/test_sample.csv")
    test_checkins = test_checkins.rename(columns={"pseudo_session_trajectory_id": sid_col})
    test_checkins[ts_col] = pd.to_datetime(test_checkins[ts_col], errors="coerce")
    test_checkins = test_checkins.sort_values([sid_col, ts_col, poi_id_col]).reset_index(drop=True)

    train_checkins = pd.read_csv(scrip_dir / f"data/{city}/train_sample.csv")
    train_checkins = train_checkins.rename(columns={"pseudo_session_trajectory_id": sid_col})
    train_checkins[ts_col] = pd.to_datetime(train_checkins[ts_col], errors="coerce")
    train_checkins = train_checkins.sort_values([sid_col, ts_col, poi_id_col]).reset_index(drop=True)

    # ── Offline: build transition index and retrieval index ──

    transition_index = build_transition_index(
        decision_state_table_df=decision_state_table_df,
        poi_descriptor_df=poi_descriptor_df,
        case_vectors=case_vectors,
        config=config,
        case_coords=case_coords,
    )

    retrieval_index = build_retrieval_index(
        case_base_df=decision_state_table_df,
        case_vectors=case_vectors,
        config=config,
        case_coords=case_coords,
    )

    # ── Step 1: Generate training data ─────────────────────────────
    # This runs both retrievers for each training decision point
    # and creates (features, label) pairs.
    # Use max_samples for faster iteration during development.

    X_train, y_train, meta_train = build_reranker_training_data(
        decision_state_table_df=decision_state_table_df,
        train_checkins_df=train_checkins,
        poi_descriptor_df=poi_descriptor_df,
        lookup_df=lookup_df,
        coord_df=coord_df,
        transition_index=transition_index,
        retrieval_index=retrieval_index,
        encoder=encoder,
        config=config,
        nearby_radius_m=2000.0,
        source_tau_m=300.0,
        recent_k=recent_k,
        query_state_source="decision_state",
        max_queries_per_session=None,
        max_samples=3000,  # start small, increase later
        train_on_pool_hits_only=True,
        max_workers=4,  # set to 4-8 on Linux for the full training-data build
        chunk_size=500,
        # mp_start_method="fork",
    )

    # ── Step 2: Train the reranker ──────────────────────────────────
    # Start with logistic regression (interpretable, fast)
    reranker = train_reranker(X_train, y_train, model_type="logistic")

    # Then try LightGBM (better at non-linear interactions):
    # reranker = train_reranker(X_train, y_train, model_type="lgbm")

    # ── Step 3: Evaluate ────────────────────────────────────────────
    metrics, details = evaluate_union_reranker(
        test_checkins_df=test_checkins,
        poi_descriptor_df=poi_descriptor_df,
        lookup_df=lookup_df,
        coord_df=coord_df,
        transition_index=transition_index,
        retrieval_index=retrieval_index,
        encoder=encoder,
        reranker=reranker,
        config=config,
        nearby_radius_m=2000.0,
        source_tau_m=300.0,
        recent_k=recent_k,
        min_checkins=3,
    )

    print("Union reranker metrics:")
    print(metrics.to_string(index=False))

    out_dir = scrip_dir / f"artifacts/{city}"
    metrics_path = out_dir / f"{city}_union_reranker_metrics.csv"
    details_path = out_dir / f"{city}_union_reranker_details.csv"
    metrics.to_csv(metrics_path, index=False)
    details.to_csv(details_path, index=False)
    cprint(f"Wrote metrics to {metrics_path}", "green")
    cprint(f"Wrote details to {details_path}", "green")
