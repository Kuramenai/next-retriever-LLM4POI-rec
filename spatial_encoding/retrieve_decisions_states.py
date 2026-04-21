from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd


def _to_query_series(query_state: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    if isinstance(query_state, pd.Series):
        return query_state
    if isinstance(query_state, pd.DataFrame):
        if len(query_state) != 1:
            raise ValueError("query_state DataFrame must contain exactly one row.")
        return query_state.iloc[0]
    raise TypeError("query_state must be a pandas Series or single-row DataFrame.")


def _safe_match_score(
    query_val,
    cand_series: pd.Series,
    weight: float,
) -> np.ndarray:
    if pd.isna(query_val):
        return np.zeros(len(cand_series), dtype=float)
    return (cand_series.astype(object).to_numpy() == query_val).astype(float) * weight


def _safe_numeric_closeness(
    query_val,
    cand_series: pd.Series,
    weight: float,
    scale: float,
) -> np.ndarray:
    if pd.isna(query_val):
        return np.zeros(len(cand_series), dtype=float)

    vals = pd.to_numeric(cand_series, errors="coerce").to_numpy(dtype=float)
    q = float(query_val)

    diff = np.abs(vals - q)
    score = np.exp(-diff / float(scale))
    score[np.isnan(score)] = 0.0
    return score * weight


def retrieve_similar_decision_states(
    query_state: Union[pd.Series, pd.DataFrame],
    case_base_df: pd.DataFrame,
    config,
    *,
    top_k: int = 50,
    same_prototype_only: bool = True,
    exclude_same_session: bool = True,
) -> pd.DataFrame:
    """
    Retrieve top-k historical decision states similar to the current query state.

    Assumes query_state and case_base_df share the decision-state schema.
    """

    q = _to_query_series(query_state)
    cands = case_base_df.copy()

    if len(cands) == 0:
        return cands.copy()

    # ------------------------------------------------------------
    # Optional filtering
    # ------------------------------------------------------------
    session_col = config.session_id_col
    proto_col = "proto_prototype_id"

    if exclude_same_session and session_col in q.index and session_col in cands.columns:
        cands = cands.loc[cands[session_col] != q[session_col]].copy()

    if same_prototype_only and proto_col in q.index and proto_col in cands.columns:
        if pd.notna(q[proto_col]):
            proto_mask = cands[proto_col] == q[proto_col]
            if proto_mask.any():
                cands = cands.loc[proto_mask].copy()

    if len(cands) == 0:
        return cands.copy()

    # ------------------------------------------------------------
    # Categorical / discrete match scores
    # ------------------------------------------------------------
    cat_weights = {
        "proto_prototype_id": 3.0,
        "current_time_bin": 1.0,
        "current_category": 2.0,
        "curr_coarse_region_token": 2.0,
        "curr_fine_region_token": 1.5,
        "curr_density_bin": 1.0,
        "curr_connectivity_bin": 1.0,
        "prev1_gap_bin": 1.5,
        "prev1_distance_bin": 1.5,
        "prev1_direction_bin": 1.0,
        "prev2_gap_bin": 0.75,
        "prev2_distance_bin": 0.75,
        "prev2_direction_bin": 0.5,
    }

    # ------------------------------------------------------------
    # Numeric closeness scores
    # ------------------------------------------------------------
    num_specs = {
        "prefix_len": {"weight": 1.0, "scale": 3.0},
        "prefix_elapsed_min": {"weight": 1.0, "scale": 60.0},
        "prefix_repeat_ratio": {"weight": 0.75, "scale": 0.25},
        "prefix_unique_poi_count": {"weight": 0.5, "scale": 3.0},
        "prefix_unique_category_count": {"weight": 0.5, "scale": 2.0},
    }

    score = np.zeros(len(cands), dtype=float)

    for col, w in cat_weights.items():
        if col in q.index and col in cands.columns:
            score += _safe_match_score(q[col], cands[col], w)

    for col, spec in num_specs.items():
        if col in q.index and col in cands.columns:
            score += _safe_numeric_closeness(
                q[col],
                cands[col],
                weight=spec["weight"],
                scale=spec["scale"],
            )

    out = cands.copy()
    out["retrieval_score"] = score

    # stable ordering
    sort_cols = ["retrieval_score"]
    ascending = [False]

    if "next_POIId" in out.columns:
        sort_cols.append("next_POIId")
        ascending.append(True)

    out = (
        out.sort_values(sort_cols, ascending=ascending)
        .head(top_k)
        .reset_index(drop=True)
    )
    return out


if __name__ == "__main__":
    # top_cases = retrieve_similar_decision_states(
    #     query_state=current_state_df,
    #     case_base_df=decision_state_df,
    #     config=config,
    #     top_k=50,
    #     same_prototype_only=True,
    # )
