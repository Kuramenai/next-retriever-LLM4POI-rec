"""
Candidate next-POI aggregation from retrieved decision states.

Pipeline:
    1. retrieve_similar_decision_states()  →  similar historical states
    2. build_candidate_next_pois()         →  weighted candidate POI ranking
    3. retrieve_candidate_next_pois()      →  convenience wrapper for 1 + 2
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from spatial_encoding.retrieve_decisions_states import DecisionStateEncoder
from spatial_encoding.retrieve_decisions_states import retrieve_similar_decision_states


def _softmax_weights(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores

    z = (scores - np.max(scores)) / float(temperature)
    w = np.exp(z)
    denom = np.sum(w)

    if denom <= 0 or np.isnan(denom):
        return np.ones_like(scores) / float(len(scores))

    return w / denom


def build_candidate_next_pois(
    retrieved_cases_df: pd.DataFrame,
    config,
    *,
    top_m: int = 20,
    score_col: str = "retrieval_score",
    temperature: float = 1.0,
) -> pd.DataFrame:
    """
    Aggregate retrieved decision states into candidate next-POI predictions.

    Parameters
    ----------
    retrieved_cases_df:
        Output of retrieve_similar_decision_states(...), containing at least:
          - next_POIId
          - retrieval_score

    top_m:
        Number of unique candidate POIs to return.

    temperature:
        Softmax temperature for converting retrieval scores into case weights.
        Lower -> sharper weighting toward top cases.
        Higher -> smoother weighting across cases.

    Returns
    -------
    candidate_df:
        One row per candidate next POI, ranked by probability.
    """
    empty_cols = [
        "next_POIId",
        "candidate_weight",
        "candidate_prob",
        "support_case_count",
        "max_case_score",
        "mean_case_score",
    ]

    if len(retrieved_cases_df) == 0:
        return pd.DataFrame(columns=empty_cols)

    if "next_POIId" not in retrieved_cases_df.columns:
        raise ValueError("retrieved_cases_df must contain 'next_POIId'")
    if score_col not in retrieved_cases_df.columns:
        raise ValueError(f"retrieved_cases_df must contain {score_col!r}")

    df = retrieved_cases_df.copy()
    df = df.loc[df["next_POIId"].notna()].copy()

    if len(df) == 0:
        return pd.DataFrame(columns=empty_cols)

    # Turn retrieval scores into normalized case weights
    df["case_weight"] = _softmax_weights(
        df[score_col].to_numpy(dtype=float), temperature
    )

    agg_dict = {
        "case_weight": "sum",
        score_col: ["count", "max", "mean"],
    }

    grouped = df.groupby("next_POIId", dropna=False).agg(agg_dict).reset_index()
    grouped.columns = [
        "next_POIId",
        "candidate_weight",
        "support_case_count",
        "max_case_score",
        "mean_case_score",
    ]

    total_weight = grouped["candidate_weight"].sum()
    if total_weight <= 0 or np.isnan(total_weight):
        grouped["candidate_prob"] = 1.0 / float(len(grouped))
    else:
        grouped["candidate_prob"] = grouped["candidate_weight"] / total_weight

    # Carry next_category if available
    if "next_category" in df.columns:
        next_cat = (
            df.groupby("next_POIId")["next_category"]
            .agg(
                lambda s: s.dropna().iloc[0]
                if s.dropna().shape[0] > 0
                else np.nan
            )
            .reset_index()
        )
        grouped = grouped.merge(next_cat, on="next_POIId", how="left")

    # Session diversity diagnostic
    if config.session_id_col in df.columns:
        session_support = (
            df.groupby("next_POIId")[config.session_id_col]
            .nunique()
            .reset_index(name="support_session_count")
        )
        grouped = grouped.merge(session_support, on="next_POIId", how="left")

    grouped = grouped.sort_values(
        ["candidate_prob", "support_case_count", "max_case_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return grouped.head(top_m)


def aggregate_candidate_pois_from_retrieved_cases(
    retrieved_cases_df: pd.DataFrame,
    *,
    config,
    top_k_candidates: int = 20,
    score_col: str = "retrieval_score",
    temperature: float = 1.0,
) -> pd.DataFrame:
    """
    Backwards-compatible wrapper used by older pipeline code.

    This maps the former naming (`aggregate_candidate_pois_from_retrieved_cases`)
    onto the current canonical implementation (`build_candidate_next_pois`).
    """
    return build_candidate_next_pois(
        retrieved_cases_df=retrieved_cases_df,
        config=config,
        top_m=top_k_candidates,
        score_col=score_col,
        temperature=temperature,
    )


def retrieve_candidate_next_pois(
    query_state,
    case_base_df: pd.DataFrame,
    encoder: DecisionStateEncoder,
    config,
    *,
    case_vectors: Optional[np.ndarray] = None,
    case_coords: Optional[np.ndarray] = None,
    top_k_cases: int = 50,
    top_m_pois: int = 20,
    same_prototype_only: bool = True,
    exclude_same_session: bool = True,
    temperature: float = 1.0,
) -> dict:
    """
    End-to-end: retrieve similar states → aggregate into candidate POIs.

    Parameters
    ----------
    query_state : Series or single-row DataFrame
        Output of build_current_decision_state().
    case_base_df : DataFrame
        Full training decision-state table (with next_POIId labels).
    encoder : DecisionStateEncoder
        Fitted encoder.
    config : SpatialEncodingConfig
    case_vectors : ndarray, optional
        Pre-computed encoder.transform(case_base_df). Shape (N, D).
    case_coords : ndarray, optional
        Pre-computed encoder.extract_coords(case_base_df). Shape (N, 2).
    top_k_cases : int
        Number of similar states to retrieve.
    top_m_pois : int
        Number of candidate POIs to return.
    same_prototype_only : bool
        Restrict to same prototype cluster.
    exclude_same_session : bool
        Exclude the query's own session.
    temperature : float
        Softmax temperature for case weighting.

    Returns
    -------
    dict with:
        - "retrieved_cases": top-k similar decision states
        - "candidate_pois": aggregated candidate next POIs
    """
    retrieved_cases = retrieve_similar_decision_states(
        query_state=query_state,
        case_base_df=case_base_df,
        encoder=encoder,
        config=config,
        case_vectors=case_vectors,
        case_coords=case_coords,
        top_k=top_k_cases,
        same_prototype_only=same_prototype_only,
        exclude_same_session=exclude_same_session,
    )

    candidate_pois = build_candidate_next_pois(
        retrieved_cases_df=retrieved_cases,
        config=config,
        top_m=top_m_pois,
        temperature=temperature,
    )

    return {
        "retrieved_cases": retrieved_cases,
        "candidate_pois": candidate_pois,
    }


if __name__ == "__main__":
    pass
