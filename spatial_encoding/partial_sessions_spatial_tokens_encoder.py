"""
Online encoder for a single partial session (prefix).

Computes transitions on-the-fly using compute_single_session_transitions(),
then delegates to the standard session encoder.

Typical usage at inference time:
    encoded = encode_partial_session_online(
        partial_session_df=current_prefix_df,
        poi_descriptor_df=poi_spatial_df,
        pair_lookup_df=pair_transition_df,   # sparse static lookup
        poi_df=poi_df,                       # for haversine fallback coords
        config=config,
    )
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from spatial_encoding.pair_transition_features_extraction import (
    compute_single_session_transitions,
    build_pair_lookup_dict,
    build_poi_coord_map,
)
from spatial_encoding.sessions_spatial_tokens_encoder import (
    encode_session_spatial_tokens,
)


def encode_partial_session_online(
    partial_session_df: pd.DataFrame,
    poi_descriptor_df: pd.DataFrame,
    pair_lookup_df: pd.DataFrame,
    poi_df: pd.DataFrame,
    config,
    *,
    session_id: Any = "__ONLINE_SESSION__",
    _pair_lookup: dict | None = None,
    _poi_coord_map: dict | None = None,
) -> dict:
    """
    Encode one observed partial session (prefix) into a spatial token sequence.

    Parameters
    ----------
    partial_session_df : DataFrame
        Observed check-ins for a single partial session.
        Required columns: config.poi_id_col, config.timestamp_col,
                          config.category_col.

    poi_descriptor_df : DataFrame
        Output of build_poi_spatial_descriptors().

    pair_lookup_df : DataFrame
        Output of build_sparse_pair_transition_lookup() (static pairwise spatial data).

    poi_df : DataFrame
        POI table with at least poi_id, lat, lon.
        Used for haversine fallback when a pair is not in the sparse lookup.

    config : SpatialEncodingConfig

    session_id : optional
        Override session ID.

    _pair_lookup : dict, optional
        Pre-built pair lookup dict. If provided, pair_lookup_df is ignored.
        Use this to avoid rebuilding the dict on every call in a serving loop.

    _poi_coord_map : dict, optional
        Pre-built POI coordinate map. Same rationale as _pair_lookup.

    Returns
    -------
    dict with fields:
        - SessionId
        - n_steps
        - poi_sequence
        - spatial_token_sequence
        - spatial_token_sequence_text
        - last_observed_poi
    """
    if partial_session_df.empty:
        raise ValueError("partial_session_df is empty; cannot encode an empty prefix.")

    required_cols = [
        config.poi_id_col,
        config.timestamp_col,
        config.category_col,
    ]
    missing = [c for c in required_cols if c not in partial_session_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in partial_session_df: {missing}")

    # ---- Validate single session ----
    work_df = partial_session_df.copy()

    if config.session_id_col in work_df.columns:
        unique_sessions = work_df[config.session_id_col].dropna().unique()
        if len(unique_sessions) > 1:
            raise ValueError(
                "partial_session_df contains multiple session IDs. "
                "This encoder expects exactly one partial session."
            )
        if len(unique_sessions) == 1:
            session_id = unique_sessions[0]

    work_df[config.session_id_col] = session_id

    # ---- Build lookup dicts (or reuse pre-built ones) ----
    pair_lookup = _pair_lookup or build_pair_lookup_dict(pair_lookup_df)
    poi_coord_map = _poi_coord_map or build_poi_coord_map(poi_df, config)

    # ---- Compute transitions for this single session ----
    session_transitions = compute_single_session_transitions(
        session_df=work_df,
        pair_lookup=pair_lookup,
        poi_coord_map=poi_coord_map,
        config=config,
        session_id=session_id,
    )

    # ---- Encode ----
    encoded_df = encode_session_spatial_tokens(
        checkins_df=work_df,
        poi_descriptor_df=poi_descriptor_df,
        session_transitions_df=session_transitions,
        config=config,
    )

    if len(encoded_df) != 1:
        raise RuntimeError(
            f"encode_partial_session_online() expected exactly one encoded session "
            f"but got {len(encoded_df)}."
        )

    result = encoded_df.iloc[0].to_dict()
    result["last_observed_poi"] = result["poi_sequence"][-1]
    return result
