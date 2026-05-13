"""
Spatial-transition candidate retriever for next-POI prediction.

Two-stage architecture:
  Stage 1 — Candidate generation (high coverage):
      Given the user's current POI and location, find all POIs that
      were historically visited as a next-step from the current POI
      or from spatially nearby POIs.

  Stage 2 — Context-aware reranking (high precision):
      For each candidate, find training decision states where that
      candidate was the actual next-POI.  Score those cases against
      the query's context (time, movement, category).  Combine with
      transition frequency and spatial proximity.

This decouples coverage (what candidates to consider) from ranking
(which candidate is most likely), solving the recall ceiling problem
of the single-stage decision-state retriever.

Usage
-----
    # Offline: build once
    transition_index = build_transition_index(
        decision_state_table_df=train_decision_states,
        poi_descriptor_df=poi_descriptors,
        config=config,
    )

    # Online: per query
    result = retrieve_via_transitions(
        query_state=query_state,
        transition_index=transition_index,
        encoder=encoder,
        config=config,
    )
    candidate_pois = result["candidate_pois"]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Any

import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from termcolor import cprint
from tqdm import tqdm

from session_decision_state_table import build_current_decision_state
from pair_transition_features_extraction import (
    build_pair_lookup_dict,
    build_poi_coord_map,
)


from extract_poi_spatial_descriptors import SpatialEncodingConfig
from retrieve_decisions_states import DecisionStateEncoder, build_retrieval_index
from retriever_evaluation import evaluate_candidate_retriever

EARTH_RADIUS_M = 6_371_008.8


# ═══════════════════════════════════════════════════════════════════════════════
# Spatial kernel
# ═══════════════════════════════════════════════════════════════════════════════


def _haversine_scalar_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return float(EARTH_RADIUS_M * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


def _haversine_one_to_many_m(
    lat1: float,
    lon1: float,
    lats2: np.ndarray,
    lons2: np.ndarray,
) -> np.ndarray:
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lats2_r = np.radians(lats2)
    lons2_r = np.radians(lons2)
    dlat = lats2_r - lat1_r
    dlon = lons2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lats2_r) * np.sin(dlon / 2.0) ** 2
    return EARTH_RADIUS_M * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


# ═══════════════════════════════════════════════════════════════════════════════
# Transition index (offline)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TransitionIndex:
    """
    Precomputed data structures for spatial-transition candidate retrieval.

    Built once from the training decision state table.
    """

    # POI locations: {poi_id: (lat, lon)}
    poi_locations: dict[int, tuple[float, float]]

    # BallTree over all POI locations (haversine metric, radians)
    poi_tree: BallTree

    # Ordered POI IDs matching the BallTree rows
    poi_ids_ordered: np.ndarray

    # Forward transition table: {current_poi: {next_poi: count}}
    forward_transitions: dict[int, dict[int, int]]

    # Session-local forward transition counts, used to remove the query
    # session from aggregate transition evidence during evaluation/training.
    session_forward_transitions: dict[Any, dict[int, dict[int, int]]]

    # Inverted index: {next_poi: np.array of case indices in decision_state_table}
    next_poi_to_case_indices: dict[int, np.ndarray]

    # Decision state vectors (unit-normalized, for context reranking)
    case_vectors_unit: np.ndarray

    # Decision state coordinates (for spatial scoring)
    case_coords: np.ndarray

    # Full decision state table (for output construction)
    decision_state_df: pd.DataFrame

    # Session IDs for exclusion
    session_ids: np.ndarray

    # Next POI locations for spatial scoring of candidates
    next_poi_locations: dict[int, tuple[float, float]]


def build_transition_index(
    decision_state_table_df: pd.DataFrame,
    poi_descriptor_df: pd.DataFrame,
    case_vectors: np.ndarray,
    config,
    *,
    case_coords: Optional[np.ndarray] = None,
) -> TransitionIndex:
    """
    Build the transition index from training data.

    Parameters
    ----------
    decision_state_table_df:
        Output of build_decision_state_table().
        Must contain: current_POIId, next_POIId, SessionId.
    poi_descriptor_df:
        POI descriptors with POI ID, Latitude, Longitude.
    case_vectors:
        Pre-encoded decision state vectors (from encoder.transform()).
    config:
        SpatialEncodingConfig.
    case_coords:
        (N, 2) array of (lat, lon) for each decision state.
    """
    cprint("Building spatial transition index...", "yellow")

    dst = decision_state_table_df

    if len(dst) != len(case_vectors):
        raise ValueError(
            f"decision_state_table_df has {len(dst)} rows but case_vectors has {len(case_vectors)} rows."
        )

    # ── POI locations ────────────────────────────────────────────────
    poi_locs: dict[int, tuple[float, float]] = {}
    for _, row in poi_descriptor_df.drop_duplicates(subset=[config.poi_id_col]).iterrows():
        pid = row[config.poi_id_col]
        poi_locs[pid] = (float(row[config.lat_col]), float(row[config.lon_col]))

    # BallTree for fast spatial neighbor queries
    poi_ids_ordered = np.array(sorted(poi_locs.keys()))
    poi_coords_rad = np.array(
        [np.radians(poi_locs[pid]) for pid in poi_ids_ordered],
        dtype=np.float64,
    )
    poi_tree = BallTree(poi_coords_rad, metric="haversine")

    # ── Forward transition table ─────────────────────────────────────
    forward: dict[int, dict[int, int]] = {}
    session_forward: dict[Any, dict[int, dict[int, int]]] = {}
    for _, row in dst.iterrows():
        curr = row["current_POIId"]
        nxt = row["next_POIId"]
        if pd.isna(curr) or pd.isna(nxt):
            continue
        sid = row[config.session_id_col]
        curr, nxt = int(curr), int(nxt)
        if curr not in forward:
            forward[curr] = {}
        forward[curr][nxt] = forward[curr].get(nxt, 0) + 1
        session_forward.setdefault(sid, {}).setdefault(curr, {})
        session_forward[sid][curr][nxt] = session_forward[sid][curr].get(nxt, 0) + 1

    # ── Inverted index: next_POI → case indices ──────────────────────
    inverted: dict[int, list[int]] = {}
    for row_pos, row in dst.iterrows():
        nxt = row["next_POIId"]
        if pd.isna(nxt):
            continue
        nxt = int(nxt)
        if nxt not in inverted:
            inverted[nxt] = []
        inverted[nxt].append(row_pos)

    inverted_np = {k: np.array(v, dtype=np.int64) for k, v in inverted.items()}

    # ── Case vectors (unit-normalized) ───────────────────────────────
    mat = np.asarray(case_vectors, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    mat_unit = mat / norms

    # ── Case coordinates ─────────────────────────────────────────────
    if case_coords is not None:
        coords = np.asarray(case_coords, dtype=np.float32)
    else:
        coords = np.full((len(dst), 2), np.nan, dtype=np.float32)

    # ── Next POI locations (for candidate spatial scoring) ───────────
    next_poi_locs: dict[int, tuple[float, float]] = {}
    for nxt_id in inverted_np.keys():
        if nxt_id in poi_locs:
            next_poi_locs[nxt_id] = poi_locs[nxt_id]

    session_ids = dst[config.session_id_col].to_numpy()

    cprint(
        f"Transition index built: {len(poi_locs)} POIs, "
        f"{len(forward)} source POIs with transitions, "
        f"{sum(len(v) for v in forward.values())} total transition entries, "
        f"{len(inverted_np)} unique next-POIs.",
        "green",
    )

    return TransitionIndex(
        poi_locations=poi_locs,
        poi_tree=poi_tree,
        poi_ids_ordered=poi_ids_ordered,
        forward_transitions=forward,
        session_forward_transitions=session_forward,
        next_poi_to_case_indices=inverted_np,
        case_vectors_unit=mat_unit,
        case_coords=coords,
        decision_state_df=dst,
        session_ids=session_ids,
        next_poi_locations=next_poi_locs,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Candidate generation (Stage 1)
# ═══════════════════════════════════════════════════════════════════════════════


def _generate_candidates_from_transitions(
    current_poi_id: int,
    query_lat: float,
    query_lon: float,
    transition_index: TransitionIndex,
    *,
    nearby_radius_m: float = 1000.0,
    max_nearby_pois: int = 50,
    source_tau_m=300.0,  # Newly added
    exclude_current: bool = True,
    exclude_session_id=None,
) -> dict[int, dict]:
    """
    Stage 1: Generate candidate next-POIs from observed transitions.

    For the current POI and its spatial neighbors, collect all historical
    next-POIs and their transition frequencies.

    Returns
    -------
    candidates: dict[next_poi_id → {
        "transition_count": int,
        "source_pois": set of current_POIs that link here,
        "from_exact_poi": bool (transition from exact current POI),
    }]
    """
    idx = transition_index
    candidates: dict[int, dict] = {}
    excluded_forward = {}
    if exclude_session_id is not None:
        excluded_forward = getattr(idx, "session_forward_transitions", {}).get(exclude_session_id, {})

    def _add_transitions(source_poi_id: int, is_exact: bool, source_weight: float = 1.0):
        transitions = idx.forward_transitions.get(source_poi_id, {})
        excluded_transitions = excluded_forward.get(source_poi_id, {})
        for next_poi, count in transitions.items():
            count = int(count) - int(excluded_transitions.get(next_poi, 0))
            if count <= 0:
                continue
            if exclude_current and next_poi == current_poi_id:
                continue
            if next_poi not in candidates:
                candidates[next_poi] = {
                    "transition_count": 0,
                    "source_pois": set(),
                    "from_exact_poi": False,
                    # Newly added: transition weight and best source weight
                    "transition_weight": 0.0,
                    "best_source_weight": 0.0,
                }
            candidates[next_poi]["transition_count"] += count
            candidates[next_poi]["source_pois"].add(source_poi_id)
            # Newly added
            candidates[next_poi]["transition_weight"] += float(count) * float(source_weight)
            candidates[next_poi]["best_source_weight"] = max(
                candidates[next_poi]["best_source_weight"], float(source_weight)
            )
            if is_exact:
                candidates[next_poi]["from_exact_poi"] = True

    # Transitions from exact current POI
    _add_transitions(int(current_poi_id), is_exact=True, source_weight=1.0)

    # Transitions from spatially nearby POIs
    if not np.isnan(query_lat) and not np.isnan(query_lon):
        query_rad = np.radians([[query_lat, query_lon]])
        radius_rad = nearby_radius_m / EARTH_RADIUS_M

        neighbor_indices, neighbor_dists = idx.poi_tree.query_radius(
            query_rad, r=radius_rad, return_distance=True, sort_results=True
        )

        neighbor_indices = neighbor_indices[0]
        neighbor_dists = neighbor_dists[0] * EARTH_RADIUS_M

        # Limit to max_nearby_pois closest
        if len(neighbor_indices) > max_nearby_pois:
            neighbor_indices = neighbor_indices[:max_nearby_pois]
            neighbor_dists = neighbor_dists[:max_nearby_pois]

        for ni, dist in zip(neighbor_indices, neighbor_dists):
            nearby_poi = int(idx.poi_ids_ordered[ni])
            if nearby_poi != current_poi_id:
                source_weight = float(np.exp(-dist / source_tau_m))
                _add_transitions(nearby_poi, is_exact=False, source_weight=source_weight)

    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# Context-aware reranking (Stage 2)
# ═══════════════════════════════════════════════════════════════════════════════


def _rerank_candidates(
    candidates: dict[int, dict],
    query_vec: np.ndarray,
    query_lat: float,
    query_lon: float,
    transition_index: TransitionIndex,
    *,
    query_session_id=None,
    exclude_same_session: bool = True,
    context_max_cases: int = 50,
    spatial_tau_m: float = 500.0,
    w_transition: float = 1.0,
    w_spatial: float = 2.0,
    w_context: float = 1.5,
    w_exact_bonus: float = 1.0,
) -> list[dict]:
    """
    Stage 2: Score each candidate using transition frequency,
    spatial proximity to current location, and context similarity.

    Score = w_transition * transition_score
          + w_spatial    * spatial_score
          + w_context    * context_score
          + w_exact      * exact_poi_bonus

    transition_score: log1p(count) / max_log_count  (normalized frequency)
    spatial_score:    exp(-haversine(current, candidate) / tau)
    context_score:    max cosine similarity between query and training cases
                      that led to this candidate
    exact_poi_bonus:  1.0 if the transition was observed from the exact current POI
    """
    idx = transition_index

    if len(candidates) == 0:
        return []

    # Precompute normalization for transition frequency
    # max_log_count = max(np.log1p(c["transition_count"]) for c in candidates.values())
    # if max_log_count < 1e-12:
    #     max_log_count = 1.0
    max_log_weight = max(np.log1p(c["transition_weight"]) for c in candidates.values())
    if max_log_weight < 1e-12:
        max_log_weight = 1.0

    # Normalize query vector
    qnorm = np.linalg.norm(query_vec)
    if qnorm > 1e-12:
        query_unit = query_vec / qnorm
    else:
        query_unit = query_vec

    scored = []

    for next_poi_id, cand_info in candidates.items():
        # ── Transition score ─────────────────────────────────────
        transition_score = np.log1p(cand_info["transition_weight"]) / max_log_weight

        # ── Spatial score ────────────────────────────────────────
        if next_poi_id in idx.next_poi_locations and not np.isnan(query_lat):
            clat, clon = idx.next_poi_locations[next_poi_id]
            dist_m = _haversine_scalar_m(query_lat, query_lon, clat, clon)
            spatial_score = float(np.exp(-dist_m / spatial_tau_m))
        else:
            spatial_score = 0.0

        # ── Context score ────────────────────────────────────────
        # Find training cases where this candidate was the next-POI
        context_score = 0.0
        case_indices = idx.next_poi_to_case_indices.get(next_poi_id)

        if case_indices is not None and len(case_indices) > 0:
            # Optionally exclude same session
            if exclude_same_session and query_session_id is not None:
                mask = idx.session_ids[case_indices] != query_session_id
                case_indices = case_indices[mask]

            # if len(case_indices) > 0:
            #     # Subsample if too many cases (for speed)
            #     if len(case_indices) > context_max_cases:
            #         case_indices = np.random.choice(case_indices, size=context_max_cases, replace=False)

            # Cosine similarity: query_unit dot each case vector
            cosines = idx.case_vectors_unit[case_indices] @ query_unit
            context_score = float(np.max(cosines))

        # ── Exact POI bonus ──────────────────────────────────────
        exact_bonus = 1.0 if cand_info["from_exact_poi"] else 0.0
        # exact_bonus = min(cand_info["exact_transition_count"] / 5.0, 1.0)

        # ── Combined score ───────────────────────────────────────
        final_score = (
            w_transition * transition_score
            + w_spatial * spatial_score
            + w_context * context_score
            + w_exact_bonus * exact_bonus
        )

        scored.append(
            {
                "next_POIId": next_poi_id,
                "final_score": final_score,
                "transition_score": transition_score,
                "spatial_score": spatial_score,
                "context_score": context_score,
                "exact_poi_bonus": exact_bonus,
                "transition_count": cand_info["transition_count"],
                "n_source_pois": len(cand_info["source_pois"]),
                "from_exact_poi": cand_info["from_exact_poi"],
            }
        )

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return scored


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


def retrieve_via_transitions(
    query_state: Union[pd.Series, pd.DataFrame],
    transition_index: TransitionIndex,
    encoder,
    config,
    *,
    top_m: int = 20,
    nearby_radius_m: float = 1000.0,
    max_nearby_pois: int = 50,
    spatial_tau_m: float = 500.0,
    context_max_cases: int = 50,
    exclude_same_session: bool = True,
    exclude_current_poi: bool = True,
    w_transition: float = 1.0,
    w_spatial: float = 2.0,
    w_context: float = 1.5,
    w_exact_bonus: float = 1.0,
    source_tau_m: float = 300.0,
    gold_poi_id: Optional[int] = None,
) -> dict:
    """
    Two-stage candidate retrieval via spatial transitions.

    Parameters
    ----------
    query_state:
        Output of build_current_decision_state().
    transition_index:
        Pre-built TransitionIndex (see build_transition_index).
    encoder:
        Fitted DecisionStateEncoder (for context vectors).
    config:
        SpatialEncodingConfig.
    top_m:
        Number of candidate POIs to return.
    nearby_radius_m:
        Radius for finding nearby source POIs in Stage 1.
    max_nearby_pois:
        Max nearby POIs to consider for transitions.
    spatial_tau_m:
        Decay parameter for spatial scoring of candidates.
    context_max_cases:
        Max training cases to evaluate per candidate for context scoring.
    w_transition, w_spatial, w_context, w_exact_bonus:
        Scoring weights for the reranking stage.

    Returns
    -------
    dict with:
        - "candidate_pois": DataFrame of ranked candidate next-POIs
        - "n_candidates_generated": total candidates before top-m cutoff
        - "n_from_exact_poi": candidates from exact current POI transitions
    """
    # Normalize query
    if isinstance(query_state, pd.DataFrame):
        if len(query_state) != 1:
            raise ValueError("query_state must be a single-row DataFrame.")
        q = query_state.iloc[0]
    elif isinstance(query_state, pd.Series):
        q = query_state
    else:
        raise TypeError("query_state must be a Series or single-row DataFrame.")

    current_poi = q.get("current_POIId")
    if pd.isna(current_poi):
        raise ValueError("query_state must have a valid current_POIId.")
    current_poi = int(current_poi)

    query_lat = float(q.get("curr_Latitude", np.nan))
    query_lon = float(q.get("curr_Longitude", np.nan))

    query_session_id = q.get(config.session_id_col, None)

    # ── Stage 1: Candidate generation ────────────────────────────────
    candidates = _generate_candidates_from_transitions(
        current_poi_id=current_poi,
        query_lat=query_lat,
        query_lon=query_lon,
        transition_index=transition_index,
        nearby_radius_m=nearby_radius_m,
        max_nearby_pois=max_nearby_pois,
        exclude_current=exclude_current_poi,
        source_tau_m=source_tau_m,
        exclude_session_id=query_session_id if exclude_same_session else None,
    )

    if not candidates:
        return {
            "candidate_pois": pd.DataFrame(
                columns=[
                    "next_POIId",
                    "final_score",
                    "transition_score",
                    "spatial_score",
                    "context_score",
                ]
            ),
            "n_candidates_generated": 0,
            "n_from_exact_poi": 0,
        }

    # ── Stage 2: Context-aware reranking ─────────────────────────────
    query_vec = np.asarray(encoder.transform_single(q), dtype=np.float32)

    scored = _rerank_candidates(
        candidates=candidates,
        query_vec=query_vec,
        query_lat=query_lat,
        query_lon=query_lon,
        transition_index=transition_index,
        query_session_id=query_session_id,
        exclude_same_session=exclude_same_session,
        context_max_cases=context_max_cases,
        spatial_tau_m=spatial_tau_m,
        w_transition=w_transition,
        w_spatial=w_spatial,
        w_context=w_context,
        w_exact_bonus=w_exact_bonus,
    )

    # ── Extract gold POI scores for diagnostics ──────────────────
    gold_poi_scores = None
    if gold_poi_id is not None:
        gold_poi_id_norm = int(gold_poi_id)
        for rank_pos, entry in enumerate(scored, start=1):
            if int(entry["next_POIId"]) == gold_poi_id_norm:
                gold_poi_scores = {
                    "gold_rank_in_full_list": rank_pos,
                    "gold_final_score": entry["final_score"],
                    "gold_transition_score": entry["transition_score"],
                    "gold_spatial_score": entry["spatial_score"],
                    "gold_context_score": entry["context_score"],
                    "gold_exact_poi_bonus": entry["exact_poi_bonus"],
                    "gold_transition_count": entry["transition_count"],
                    "gold_from_exact_poi": entry["from_exact_poi"],
                }
                break

        # Also get the 20th-ranked candidate for comparison
        if len(scored) >= 20:
            rank20 = scored[19]
            if gold_poi_scores is None:
                gold_poi_scores = {"gold_rank_in_full_list": None}
            gold_poi_scores["rank20_final_score"] = rank20["final_score"]
            gold_poi_scores["rank20_transition_score"] = rank20["transition_score"]
            gold_poi_scores["rank20_spatial_score"] = rank20["spatial_score"]
            gold_poi_scores["rank20_context_score"] = rank20["context_score"]

    candidate_df = pd.DataFrame(scored[:top_m])

    # Add next_category if available
    idx = transition_index
    if "next_category" in idx.decision_state_df.columns:
        cat_map = (
            idx.decision_state_df.dropna(subset=["next_POIId"])
            .drop_duplicates(subset=["next_POIId"], keep="first")
            .set_index("next_POIId")["next_category"]
            .to_dict()
        )
        candidate_df["next_category"] = candidate_df["next_POIId"].map(cat_map)

    n_from_exact = sum(1 for c in candidates.values() if c["from_exact_poi"])

    return {
        "candidate_pois": candidate_df,
        "n_candidates_generated": len(candidates),
        "n_from_exact_poi": n_from_exact,
        "generated_candidate_ids": list(candidates.keys()),
        "gold_poi_scores": gold_poi_scores,  # ← add this
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation helper (compatible with existing evaluate_candidate_retriever)
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_transition_retriever(
    test_checkins_df: pd.DataFrame,
    *,
    poi_descriptor_df: pd.DataFrame,
    lookup_df: pd.DataFrame,
    coord_df: pd.DataFrame,
    transition_index: TransitionIndex,
    encoder,
    config,
    k_values: tuple[int, ...] = (1, 3, 5, 10, 20),
    top_m_pois: int = 20,
    nearby_radius_m: float = 1000.0,
    spatial_tau_m: float = 500.0,
    context_max_cases: int = 50,
    w_transition: float = 1.0,
    w_spatial: float = 2.0,
    w_context: float = 1.5,
    w_exact_bonus: float = 1.0,
    source_tau_m: float = 300.0,
    min_checkins: int = 2,
    recent_k: Optional[int] = None,
    max_sessions: Optional[int] = None,
    random_state: int = 42,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate the transition-based retriever.

    Same interface as evaluate_candidate_retriever for easy comparison.
    """

    def _normalize_poi_id(value) -> Any:
        if pd.isna(value):
            return None
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return int(value) if float(value).is_integer() else float(value)
        return str(value)

    def _rank_of_gold(candidate_ids, gold_poi_id) -> Optional[int]:
        gold = _normalize_poi_id(gold_poi_id)
        for i, cid in enumerate(candidate_ids, start=1):
            if _normalize_poi_id(cid) == gold:
                return i
        return None

    k_values = tuple(sorted({int(k) for k in k_values if int(k) > 0}))
    top_m_pois = max(top_m_pois, max(k_values))

    encoder_recent_k = int(getattr(encoder, "recent_k", 2))
    if recent_k is None:
        recent_k = encoder_recent_k

    work = test_checkins_df.copy()
    work[config.timestamp_col] = pd.to_datetime(work[config.timestamp_col], errors="coerce")
    work = work.loc[work[config.timestamp_col].notna()].copy()
    sort_cols = [config.session_id_col, config.timestamp_col, config.poi_id_col]
    work = work.sort_values(sort_cols).reset_index(drop=True)

    session_ids = work[config.session_id_col].drop_duplicates().tolist()
    if max_sessions is not None and len(session_ids) > int(max_sessions):
        rng = np.random.default_rng(int(random_state))
        session_ids = rng.choice(session_ids, size=int(max_sessions), replace=False).tolist()
        work = work.loc[work[config.session_id_col].isin(set(session_ids))].copy()

    groups = work.groupby(config.session_id_col, sort=False)
    iterator = tqdm(groups, desc="evaluate transition retriever", unit="session") if show_progress else groups

    rows: list[dict] = []

    for session_id, session_df in iterator:
        session_df = session_df.sort_values([config.timestamp_col, config.poi_id_col]).reset_index(drop=True)

        base_row = {
            config.session_id_col: session_id,
            "n_checkins": int(len(session_df)),
            "skipped": False,
            "error": None,
        }

        if len(session_df) < int(min_checkins):
            rows.append(
                {
                    **base_row,
                    "skipped": True,
                    "error": f"fewer than {min_checkins} check-ins",
                    "gold_next_POIId": None,
                    "gold_rank": None,
                    "candidate_count": 0,
                    "n_candidates_generated": 0,
                    "n_from_exact_poi": 0,
                }
            )
            continue

        prefix_df = session_df.iloc[:-1].copy().reset_index(drop=True)
        gold_next = session_df.iloc[-1]
        gold_poi_id = gold_next[config.poi_id_col]

        try:
            query_state = build_current_decision_state(
                partial_session_df=prefix_df,
                poi_descriptor_df=poi_descriptor_df,
                config=config,
                lookup_df=lookup_df,
                coord_df=coord_df,
                recent_k=recent_k,
            )

            result = retrieve_via_transitions(
                query_state=query_state,
                transition_index=transition_index,
                encoder=encoder,
                config=config,
                top_m=top_m_pois,
                nearby_radius_m=nearby_radius_m,
                spatial_tau_m=spatial_tau_m,
                context_max_cases=context_max_cases,
                exclude_same_session=True,
                w_transition=w_transition,
                w_spatial=w_spatial,
                w_context=w_context,
                w_exact_bonus=w_exact_bonus,
                source_tau_m=source_tau_m,
                gold_poi_id=int(gold_poi_id),
            )

            candidate_pois = result["candidate_pois"]
            candidate_ids = candidate_pois["next_POIId"].tolist() if len(candidate_pois) > 0 else []
            rank = _rank_of_gold(candidate_ids, gold_poi_id)
            generated_ids = result["generated_candidate_ids"]

            rec = {
                **base_row,
                "gold_next_POIId": gold_poi_id,
                "gold_rank": rank,
                "candidate_count": int(len(candidate_pois)),
                "n_candidates_generated": result["n_candidates_generated"],
                "n_from_exact_poi": result["n_from_exact_poi"],
                "top_candidates": candidate_ids[: max(k_values)],
                "generated_pool_hit": _normalize_poi_id(gold_poi_id)
                in {_normalize_poi_id(x) for x in generated_ids},
            }
            # Merge gold POI diagnostic scores
            if result.get("gold_poi_scores"):
                rec.update(result["gold_poi_scores"])
            for k in k_values:
                rec[f"hit@{k}"] = bool(rank is not None and rank <= k)
                rec[f"recall@{k}"] = float(rec[f"hit@{k}"])
            rows.append(rec)

        except Exception as e:
            rec = {
                **base_row,
                "error": repr(e),
                "gold_next_POIId": gold_poi_id,
                "gold_rank": None,
                "candidate_count": 0,
                "n_candidates_generated": 0,
                "n_from_exact_poi": 0,
                "generated_pool_hit": None,
            }
            for k in k_values:
                rec[f"hit@{k}"] = False
                rec[f"recall@{k}"] = 0.0
            rows.append(rec)

    details_df = pd.DataFrame(rows)

    # Build summary
    valid_mask = (~details_df["skipped"].fillna(False)) & details_df["error"].isna()
    valid = details_df.loc[valid_mask].copy()

    summary: dict = {
        "n_sessions_total": int(len(details_df)),
        "n_sessions_evaluated": int(len(valid)),
        "n_sessions_skipped": int(details_df["skipped"].fillna(False).sum()),
    }

    if len(valid) > 0:
        ranks = pd.to_numeric(valid["gold_rank"], errors="coerce")
        for k in k_values:
            summary[f"hit@{k}"] = float(valid[f"hit@{k}"].mean())
            summary[f"recall@{k}"] = summary[f"hit@{k}"]
        summary["mrr"] = float((1.0 / ranks.dropna()).sum() / len(valid))
        summary["mean_gold_rank"] = float(ranks.dropna().mean()) if ranks.notna().any() else np.nan
        summary["mean_candidates_generated"] = float(valid["n_candidates_generated"].mean())
        summary["mean_from_exact_poi"] = float(valid["n_from_exact_poi"].mean())
        summary["generated_pool_recall"] = float(valid["generated_pool_hit"].mean())
        summary["reranking_loss@20"] = summary["generated_pool_recall"] - summary["recall@20"]
    else:
        for k in k_values:
            summary[f"hit@{k}"] = np.nan
            summary[f"recall@{k}"] = np.nan
        summary["mrr"] = np.nan
        summary["mean_gold_rank"] = np.nan
        summary["generated_pool_recall"] = np.nan
        summary["reranking_loss@20"] = np.nan

    return pd.DataFrame([summary]), details_df


def _hit_mask(s: pd.Series) -> pd.Series:
    """
    Coerce hit-like columns (bool, 0/1 int/float, NaN) to a boolean Series for ~, &, |.

    Merged frames often promote bools to float64; unary ~ is not defined on float.
    """
    num = pd.to_numeric(s, errors="coerce")
    return (num > 0).fillna(False)


def source_complementarity(ds_details, st_details, session_col="SessionId"):
    ds = ds_details[[session_col, "hit@20"]].rename(columns={"hit@20": "ds_hit20"})

    st = st_details[[session_col, "generated_pool_hit", "hit@20"]].rename(
        columns={
            "generated_pool_hit": "st_pool_hit",
            "hit@20": "st_hit20",
        }
    )

    df = ds.merge(st, on=session_col, how="inner")

    ds_h = _hit_mask(df["ds_hit20"])
    st_pool = _hit_mask(df["st_pool_hit"])
    st_h20 = _hit_mask(df["st_hit20"])

    out = {
        "n": len(df),
        "ds_hit20": float(ds_h.mean()),
        "st_pool_recall": float(st_pool.mean()),
        "st_hit20": float(st_h20.mean()),
        "both_ds_and_st_pool": float((ds_h & st_pool).mean()),
        "ds_only": float((ds_h & ~st_pool).mean()),
        "st_pool_only": float((~ds_h & st_pool).mean()),
        "neither": float((~ds_h & ~st_pool).mean()),
        "union_oracle": float((ds_h | st_pool).mean()),
    }

    return out


def analyze_reranking_failure(details_df: pd.DataFrame, top_k: int = 20):
    """
    Analyze why the reranker fails: which scoring component is responsible
    for the gold POI being ranked below top_k?
    """
    valid = details_df[(~details_df["skipped"].fillna(False)) & details_df["error"].isna()].copy()

    # Cases where gold was in pool but missed top-k
    missed = valid[
        valid["generated_pool_hit"].astype(bool)
        & (valid["gold_rank"].isna() | (valid["gold_rank"] > top_k))
        & valid["gold_final_score"].notna()
    ].copy()

    found = valid[
        valid["gold_rank"].notna() & (valid["gold_rank"] <= top_k) & valid["gold_final_score"].notna()
    ].copy()

    print(f"{'═' * 70}")
    print(f"  RERANKING FAILURE ANALYSIS (top-{top_k})")
    print(f"{'═' * 70}")
    print(f"  Found (gold in top-{top_k}):  {len(found)}")
    print(f"  Missed (gold generated but ranked >{top_k}):  {len(missed)}")
    print()

    if len(missed) == 0:
        print("  No missed cases with gold scores available.")
        return

    # Compare average scores: found vs missed
    score_cols = [
        "gold_transition_score",
        "gold_spatial_score",
        "gold_context_score",
        "gold_exact_poi_bonus",
    ]

    print(f"  {'Component':<25s} {'Found (mean)':>12s} {'Missed (mean)':>13s} {'Gap':>10s}")
    print(f"  {'─' * 62}")
    for col in score_cols:
        if col in found.columns and col in missed.columns:
            f_mean = found[col].mean()
            m_mean = missed[col].mean()
            print(f"  {col:<25s} {f_mean:>12.4f} {m_mean:>13.4f} {f_mean - m_mean:>10.4f}")

    print("\nGold vs rank-20 threshold (missed cases only):")
    print(f"  {'Component':<25s} {'Gold (mean)':>12s} {'Rank20 (mean)':>13s} {'Deficit':>10s}")
    print(f"  {'─' * 62}")

    rank20_cols = {
        "gold_transition_score": "rank20_transition_score",
        "gold_spatial_score": "rank20_spatial_score",
        "gold_context_score": "rank20_context_score",
    }
    for gold_col, r20_col in rank20_cols.items():
        if gold_col in missed.columns and r20_col in missed.columns:
            g_mean = missed[gold_col].mean()
            r_mean = missed[r20_col].mean()
            print(f"  {gold_col:<25s} {g_mean:>12.4f} {r_mean:>13.4f} {g_mean - r_mean:>10.4f}")

    # What fraction of missed cases had gold_from_exact_poi = True?
    if "gold_from_exact_poi" in missed.columns:
        exact_frac = missed["gold_from_exact_poi"].mean()
        print(f"\n  Missed cases from exact POI: {exact_frac:.1%}")
        print(f"  Found cases from exact POI:  {found['gold_from_exact_poi'].mean():.1%}")

    print(f"{'═' * 70}")


if __name__ == "__main__":
    city = "nyc"
    # max_eval_sessions = 200
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

    # ── Offline: build transition index (reuses your existing artifacts) ──

    transition_index = build_transition_index(
        decision_state_table_df=decision_state_table_df,
        poi_descriptor_df=poi_descriptor_df,
        case_vectors=case_vectors,
        config=config,
        case_coords=case_coords,
    )

    # ── Evaluate ──
    st_metrics_df, st_details_df = evaluate_transition_retriever(
        test_checkins_df=test_checkins,
        poi_descriptor_df=poi_descriptor_df,
        lookup_df=lookup_df,
        coord_df=coord_df,
        transition_index=transition_index,
        encoder=encoder,
        config=config,
        k_values=(1, 3, 5, 10, 20),
        top_m_pois=20,
        nearby_radius_m=2000.0,  # 1km radius for finding source POIs
        spatial_tau_m=500.0,
        source_tau_m=300.0,  # decay for candidate distance scoring
        recent_k=recent_k,
        context_max_cases=50,
        min_checkins=3,
        w_spatial=1.0,
    )

    print("\nTransition retriever metrics:")
    print(st_metrics_df.to_string(index=False))
    # After running evaluate_transition_retriever:
    valid = st_details_df[~st_details_df["skipped"].fillna(False) & st_details_df["error"].isna()]

    if len(valid) == 0:
        cprint(
            "No valid sessions to evaluate (all skipped or error). Metrics are NaN because `valid` is empty.",
            "red",
        )
        if "error" in st_details_df.columns:
            cprint("Top errors:", "yellow")
            print(st_details_df.loc[st_details_df["error"].notna(), "error"].value_counts().head(10))
        raise SystemExit(1)

    # 1. How many candidates is Stage 1 generating?
    print(f"Mean candidates generated: {valid['n_candidates_generated'].mean():.1f}")
    print(f"Median: {valid['n_candidates_generated'].median():.0f}")
    print(f"Min: {valid['n_candidates_generated'].min()}")
    print(f"Max: {valid['n_candidates_generated'].max()}")

    # 2. How often is the gold POI in the generated candidate set?
    # (Stage 1 coverage — this is the new ceiling)
    print(f"\nMean from exact POI: {valid['n_from_exact_poi'].mean():.1f}")

    # 3. Is the gold POI being generated but ranked poorly,
    #    or not generated at all?
    gold_in_top50 = valid["gold_rank"].notna().mean()
    print(
        f"\nGold in top-20 (ranked): {(valid['gold_rank'].notna() & (valid['gold_rank'] <= 20)).mean():.3f}"
    )
    print(f"Gold in top-50 (ranked): {gold_in_top50:.3f}")

    out_dir = scrip_dir / f"artifacts/{city}"
    metrics_path = out_dir / f"{city}_transition_retriever_metrics.csv"
    details_path = out_dir / f"{city}_transition_retriever_details.csv"
    st_metrics_df.to_csv(metrics_path, index=False)
    st_details_df[~st_details_df["skipped"].fillna(False) & st_details_df["error"].isna()].to_csv(
        details_path, index=False
    )
    cprint(f"Wrote metrics to {metrics_path}", "green")
    cprint(f"Wrote details to {details_path}", "green")

    analyze_reranking_failure(st_details_df, top_k=20)

    ###################################################################################

    # retrieval_index = build_retrieval_index(
    #     case_base_df=decision_state_table_df,
    #     case_vectors=case_vectors,
    #     config=config,
    #     case_coords=case_coords,
    # )

    # ds_metrics_df, ds_details_df = evaluate_candidate_retriever(
    #     test_checkins_df=test_checkins,
    #     poi_descriptor_df=poi_descriptor_df,
    #     lookup_df=lookup_df,
    #     coord_df=coord_df,
    #     prototype_assignments_df=None,
    #     encoder=encoder,
    #     retrieval_index=retrieval_index,
    #     config=config,
    #     k_values=k_values,
    #     top_k_cases=50,
    #     top_m_pois=max(k_values),
    #     temperature=0.2,
    #     max_sessions=None,
    #     show_progress=True,
    #     same_prototype_only=False,
    #     exclude_same_session=True,
    #     prototype_union_k=3,
    #     recent_k=recent_k,
    #     min_checkins=3,
    # )

    # print(source_complementarity(ds_details_df, st_details_df))
