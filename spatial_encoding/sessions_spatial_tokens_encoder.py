"""
Session-level spatial token encoder.

Consumes pre-computed transitions (from pair_transition_features_extraction)
and per-POI descriptors. The encoder is a pure token assembler — no spatial
math, no temporal binning, just dictionary lookups.

Expected workflow:
    1. build_all_session_transition_descriptors()  →  session_transitions_df
    2. encode_session_spatial_tokens(checkins_df, poi_descriptor_df,
                                     session_transitions_df, config)
"""

from __future__ import annotations

from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _time_of_day_bin(ts: pd.Timestamp) -> str:
    """Four-way coarse time-of-day bin."""
    hour = ts.hour
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 22:
        return "evening"
    else:
        return "late_night"


def _serialize_spatial_token(token: dict[str, Any]) -> str:
    """
    Compact string serialization for retrieval indexing and prompt compression.

    NOTE: Update this if you add/remove token fields. If you need a separate
    prompt-facing format (natural language summary), write a second serializer
    rather than overloading this one.
    """
    parts = [
        f"T:{token['time_bin']}",
        f"CAT:{token['poi_category']}",
        f"R8:{token['region_coarse_token']}",
        f"R9:{token['region_fine_token']}",
        f"DIST:{token['distance_bin']}",
        f"DIR:{token['direction_bin']}",
        f"GAP:{token['gap_bin']}",
        f"DEN:{token['density_bin']}",
        f"CONN:{token['connectivity_bin']}",
    ]
    return "|".join(parts)


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

def encode_session_spatial_tokens(
    checkins_df: pd.DataFrame,
    poi_descriptor_df: pd.DataFrame,
    session_transitions_df: pd.DataFrame,
    config,
) -> pd.DataFrame:
    """
    Encode each session into a sequence of spatial tokens.

    All spatial and temporal features come from pre-computed lookups.
    This function only reads from dicts — no haversine, no gap binning,
    no pair-transition fallback.

    Parameters
    ----------
    checkins_df : DataFrame
        Raw check-ins. Required columns:
        - config.session_id_col
        - config.poi_id_col
        - config.timestamp_col
        - config.category_col

    poi_descriptor_df : DataFrame
        Output of build_poi_spatial_descriptors(). Required columns:
        - config.poi_id_col
        - region_coarse_token, region_fine_token
        - density_bin, connectivity_bin

    session_transitions_df : DataFrame
        Output of build_all_session_transition_descriptors() or
        compute_single_session_transitions(). Required columns:
        - config.session_id_col
        - transition_index
        - gap_bin, distance_bin, direction_bin

    config : SpatialEncodingConfig

    Returns
    -------
    session_tokens_df : DataFrame
        One row per session with:
        - SessionId
        - n_steps
        - poi_sequence            (list[poi_id])
        - spatial_token_sequence  (list[dict])
        - spatial_token_sequence_text  (list[str])
    """
    # ------------------------------------------------------------------
    # Validate check-in columns
    # ------------------------------------------------------------------
    required_checkin_cols = [
        config.session_id_col,
        config.poi_id_col,
        config.timestamp_col,
        config.category_col,
    ]
    missing = [c for c in required_checkin_cols if c not in checkins_df.columns]
    if missing:
        raise ValueError(f"Missing required check-in columns: {missing}")

    # ------------------------------------------------------------------
    # Validate POI descriptor columns
    # ------------------------------------------------------------------
    poi_required = [
        config.poi_id_col,
        "region_coarse_token",
        "region_fine_token",
        "density_bin",
        "connectivity_bin",
    ]
    missing_poi = [c for c in poi_required if c not in poi_descriptor_df.columns]
    if missing_poi:
        raise ValueError(f"Missing required POI descriptor columns: {missing_poi}")

    # ------------------------------------------------------------------
    # Validate transition columns
    # ------------------------------------------------------------------
    trans_required = [
        config.session_id_col,
        "transition_index",
        "gap_bin",
        "distance_bin",
        "direction_bin",
    ]
    missing_trans = [c for c in trans_required if c not in session_transitions_df.columns]
    if missing_trans:
        raise ValueError(
            f"Missing required transition columns: {missing_trans}. "
            f"Did you pass the output of build_all_session_transition_descriptors()?"
        )

    # ------------------------------------------------------------------
    # Sort checkins
    # ------------------------------------------------------------------
    df = checkins_df.copy()
    df[config.timestamp_col] = pd.to_datetime(df[config.timestamp_col])
    df = df.sort_values(
        [config.session_id_col, config.timestamp_col]
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Build fast lookups
    # ------------------------------------------------------------------
    poi_lookup: dict = (
        poi_descriptor_df[poi_required]
        .drop_duplicates(subset=[config.poi_id_col])
        .set_index(config.poi_id_col)
        .to_dict(orient="index")
    )

    # Key: (session_id, transition_index) → {gap_bin, distance_bin, direction_bin}
    transition_lookup: dict = (
        session_transitions_df[trans_required]
        .drop_duplicates(
            subset=[config.session_id_col, "transition_index"], keep="first"
        )
        .set_index([config.session_id_col, "transition_index"])
        [["gap_bin", "distance_bin", "direction_bin"]]
        .to_dict(orient="index")
    )

    # ------------------------------------------------------------------
    # Encode sessions
    # ------------------------------------------------------------------
    session_records: list[dict] = []

    for session_id, session_df in df.groupby(config.session_id_col, sort=False):
        session_df = session_df.sort_values(config.timestamp_col).reset_index(
            drop=True
        )

        poi_sequence: list = []
        token_sequence: list[dict] = []
        token_sequence_text: list[str] = []

        for pos in range(len(session_df)):
            row = session_df.iloc[pos]
            curr_poi = row[config.poi_id_col]
            curr_ts = row[config.timestamp_col]
            curr_cat = row[config.category_col]

            # ---- POI descriptor ----
            poi_desc = poi_lookup.get(curr_poi)
            if poi_desc is None:
                raise KeyError(
                    f"POI {curr_poi} (session {session_id}, position {pos}) "
                    f"is missing in poi_descriptor_df."
                )

            # ---- Transition features ----
            if pos == 0:
                gap_bin = "BOS"
                distance_bin = "BOS"
                direction_bin = "BOS"
            else:
                trans_key = (session_id, pos - 1)
                trans = transition_lookup.get(trans_key)
                if trans is None:
                    raise KeyError(
                        f"Missing pre-computed transition for "
                        f"session={session_id}, transition_index={pos - 1}. "
                        f"Ensure session_transitions_df covers all sessions "
                        f"in checkins_df."
                    )
                gap_bin = trans["gap_bin"]
                distance_bin = trans["distance_bin"]
                direction_bin = trans["direction_bin"]

            # ---- Assemble token ----
            token = {
                "time_bin": _time_of_day_bin(curr_ts),
                "poi_category": curr_cat,
                "region_coarse_token": poi_desc["region_coarse_token"],
                "region_fine_token": poi_desc["region_fine_token"],
                "distance_bin": distance_bin,
                "direction_bin": direction_bin,
                "gap_bin": gap_bin,
                "density_bin": poi_desc["density_bin"],
                "connectivity_bin": poi_desc["connectivity_bin"],
            }

            poi_sequence.append(curr_poi)
            token_sequence.append(token)
            token_sequence_text.append(_serialize_spatial_token(token))

        session_records.append(
            {
                config.session_id_col: session_id,
                "n_steps": len(session_df),
                "poi_sequence": poi_sequence,
                "spatial_token_sequence": token_sequence,
                "spatial_token_sequence_text": token_sequence_text,
            }
        )

    return pd.DataFrame(session_records)
