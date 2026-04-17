from __future__ import annotations

from typing import Any
import pandas as pd

from sessions_spatial_tokens_encoder import encode_session_spatial_tokens


def encode_partial_session_online(
    partial_session_df: pd.DataFrame,
    poi_descriptor_df: pd.DataFrame,
    pair_transition_df: pd.DataFrame,
    config,
    *,
    session_id: Any = "__ONLINE_SESSION__",
) -> dict:
    """
    Encode one observed partial session (prefix) into a spatial token sequence.

    Parameters
    ----------
    partial_session_df:
        DataFrame containing the observed check-ins for a single partial session.
        Expected columns:
        - config.poi_id_col
        - config.timestamp_col
        - config.category_col
        - config.session_id_col (optional; will be added if missing)

    poi_descriptor_df:
        Output of build_poi_spatial_descriptors(...)

    pair_transition_df:
        Output of build_sparse_pair_transition_lookup(...)

    config:
        SpatialEncodingConfig

    session_id:
        Temporary session ID used when partial_session_df does not already contain one.

    Returns
    -------
    encoded_prefix:
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

    work_df = partial_session_df.copy()

    # Ensure exactly one session identity
    if config.session_id_col not in work_df.columns:
        work_df[config.session_id_col] = session_id
    else:
        unique_sessions = work_df[config.session_id_col].dropna().unique()
        if len(unique_sessions) > 1:
            raise ValueError(
                "partial_session_df contains multiple session IDs. "
                "This encoder expects exactly one partial session."
            )
        elif len(unique_sessions) == 0:
            work_df[config.session_id_col] = session_id

    # Reuse the full session encoder
    encoded_df = encode_session_spatial_tokens(
        checkins_df=work_df,
        poi_descriptor_df=poi_descriptor_df,
        pair_transition_df=pair_transition_df,
        config=config,
    )

    if len(encoded_df) != 1:
        raise RuntimeError(
            "encode_partial_session_online(...) expected exactly one encoded session "
            f"but got {len(encoded_df)}."
        )

    result = encoded_df.iloc[0].to_dict()
    result["last_observed_poi"] = result["poi_sequence"][-1]

    return result


# encoded_prefix = encode_partial_session_online(
#     partial_session_df=current_prefix_df,
#     poi_descriptor_df=poi_spatial_df,
#     pair_transition_df=pair_transition_df,
#     config=config,
# )

# encoded_prefix["spatial_token_sequence_text"]

# test_session_prefix_df = test_session_df.sort_values(config.timestamp_col).iloc[:-1].copy()

# encoded_prefix = encode_partial_session_online(
#     partial_session_df=test_session_prefix_df,
#     poi_descriptor_df=poi_spatial_df,
#     pair_transition_df=pair_transition_df,
#     config=config,
# )
