from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from spatial_encoding.sparse_pair_transition_lookup import (
    _haversine_from_one_to_many_m,
    _bearing_from_one_to_many_deg,
    _bin_distances_m,
    _bearing_deg_to_direction_bin,
)


def _time_of_day_bin(ts: pd.Timestamp) -> str:
    """
    Four-way coarse time-of-day bin.
    """
    hour = ts.hour

    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 22:
        return "evening"
    else:
        return "late_night"


def _gap_bin_minutes(gap_min: float, edges_min: tuple[float, ...]) -> str:
    """
    Fixed absolute binning for inter-check-in time gaps in minutes.
    """
    if pd.isna(gap_min):
        return "BOS"

    lower = 0
    for edge in edges_min:
        if gap_min <= edge:
            return f"{int(lower)}-{int(edge)}min"
        lower = edge

    return f"{int(edges_min[-1])}+min"


def _serialize_spatial_token(token: dict[str, Any]) -> str:
    """
    Compact string serialization for debugging / prompt-side compression.
    """
    return (
        f"T:{token['time_bin']}"
        f"|CAT:{token['poi_category']}"
        f"|R8:{token['region_coarse_token']}"
        f"|R9:{token['region_fine_token']}"
        f"|DIST:{token['distance_bin']}"
        f"|DIR:{token['direction_bin']}"
        f"|GAP:{token['gap_bin']}"
        f"|DEN:{token['density_bin']}"
        f"|CTR:{token['centrality_bin']}"
        f"|CTX:{token['urban_context_token']}"
    )


def encode_session_spatial_tokens(
    checkins_df: pd.DataFrame,
    poi_descriptor_df: pd.DataFrame,
    pair_transition_df: pd.DataFrame,
    config,
) -> pd.DataFrame:
    """
    Encode each session into a sequence of spatial tokens.

    Expected checkins_df columns
    ----------------------------
    - config.session_id_col
    - config.poi_id_col
    - config.timestamp_col
    - config.category_col

    Expected poi_descriptor_df columns
    ----------------------------------
    - config.poi_id_col
    - region_coarse_token
    - region_fine_token
    - density_bin
    - centrality_bin
    - urban_context_token
    - Latitude / Longitude (via config.lat_col / config.lon_col) for fallback bearing/haversine

    Expected pair_transition_df columns
    -----------------------------------
    - src_POIId
    - dst_POIId
    - distance_bin
    - direction_bin
    - haversine_distance_m
    - bearing_deg
    - used_haversine_fallback

    Returns
    -------
    session_tokens_df:
        One row per session with:
        - SessionId
        - n_steps
        - poi_sequence
        - spatial_token_sequence   (list[dict])
        - spatial_token_sequence_text (list[str])
    """
    required_checkin_cols = [
        config.session_id_col,
        config.poi_id_col,
        config.timestamp_col,
        config.category_col,
    ]
    missing_checkin_cols = [
        c for c in required_checkin_cols if c not in checkins_df.columns
    ]
    if missing_checkin_cols:
        raise ValueError(f"Missing required check-in columns: {missing_checkin_cols}")

    # ---- Sort and normalize timestamps ----
    df = checkins_df.copy()
    df[config.timestamp_col] = pd.to_datetime(df[config.timestamp_col])
    df = df.sort_values([config.session_id_col, config.timestamp_col]).reset_index(
        drop=True
    )

    # ---- Build POI descriptor lookup ----
    poi_required_cols = [
        config.poi_id_col,
        "region_coarse_token",
        "region_fine_token",
        "density_bin",
        "centrality_bin",
        "urban_context_token",
        config.lat_col,
        config.lon_col,
    ]
    missing_poi_cols = [
        c for c in poi_required_cols if c not in poi_descriptor_df.columns
    ]
    if missing_poi_cols:
        raise ValueError(f"Missing required POI descriptor columns: {missing_poi_cols}")

    poi_lookup = (
        poi_descriptor_df[poi_required_cols]
        .drop_duplicates(subset=[config.poi_id_col])
        .set_index(config.poi_id_col)
        .to_dict(orient="index")
    )

    # ---- Build sparse pair lookup ----
    pair_required_cols = [
        "src_POIId",
        "dst_POIId",
        "distance_bin",
        "direction_bin",
    ]
    missing_pair_cols = [
        c for c in pair_required_cols if c not in pair_transition_df.columns
    ]
    if missing_pair_cols:
        raise ValueError(f"Missing required pair lookup columns: {missing_pair_cols}")

    pair_lookup = (
        pair_transition_df.drop_duplicates(subset=["src_POIId", "dst_POIId"])
        .set_index(["src_POIId", "dst_POIId"])[["distance_bin", "direction_bin"]]
        .to_dict(orient="index")
    )

    session_records = []

    for session_id, session_df in df.groupby(config.session_id_col, sort=False):
        session_df = session_df.sort_values(config.timestamp_col).reset_index(drop=True)

        poi_sequence = session_df[config.poi_id_col].tolist()
        token_sequence = []
        token_sequence_text = []

        for idx, row in session_df.iterrows():
            curr_poi = row[config.poi_id_col]
            curr_ts = row[config.timestamp_col]
            curr_cat = row[config.category_col]

            if curr_poi not in poi_lookup:
                raise KeyError(
                    f"POI {curr_poi} from session {session_id} is missing in poi_descriptor_df."
                )

            poi_desc = poi_lookup[curr_poi]

            # ---- Incoming edge features ----
            if idx == 0:
                distance_bin = "BOS"
                direction_bin = "BOS"
                gap_bin = "BOS"
            else:
                prev_row = session_df.iloc[idx - 1]
                prev_poi = prev_row[config.poi_id_col]
                prev_ts = prev_row[config.timestamp_col]

                gap_min = (curr_ts - prev_ts).total_seconds() / 60.0
                gap_bin = _gap_bin_minutes(gap_min, tuple(config.gap_bin_edges_min))

                pair_key = (prev_poi, curr_poi)
                pair_desc = pair_lookup.get(pair_key, None)

                if pair_desc is not None:
                    distance_bin = pair_desc["distance_bin"]
                    direction_bin = pair_desc["direction_bin"]
                else:
                    # Fallback if sparse pair lookup misses this transition
                    prev_desc = poi_lookup.get(prev_poi)
                    if prev_desc is None:
                        raise KeyError(
                            f"Previous POI {prev_poi} from session {session_id} "
                            f"is missing in poi_descriptor_df."
                        )

                    fallback_dist_m = _haversine_from_one_to_many_m(
                        lat1=float(prev_desc[config.lat_col]),
                        lon1=float(prev_desc[config.lon_col]),
                        lats2=np.array([float(poi_desc[config.lat_col])]),
                        lons2=np.array([float(poi_desc[config.lon_col])]),
                    )[0]

                    fallback_bearing_deg = _bearing_from_one_to_many_deg(
                        lat1=float(prev_desc[config.lat_col]),
                        lon1=float(prev_desc[config.lon_col]),
                        lats2=np.array([float(poi_desc[config.lat_col])]),
                        lons2=np.array([float(poi_desc[config.lon_col])]),
                    )[0]

                    distance_bin = _bin_distances_m(
                        np.array([fallback_dist_m]),
                        edges_m=tuple(config.distance_bin_edges_m),
                    )[0]
                    direction_bin = _bearing_deg_to_direction_bin(
                        np.array([fallback_bearing_deg])
                    )[0]

            token = {
                "time_bin": _time_of_day_bin(curr_ts),
                "poi_category": curr_cat,
                "region_coarse_token": poi_desc["region_coarse_token"],
                "region_fine_token": poi_desc["region_fine_token"],
                "distance_bin": distance_bin,
                "direction_bin": direction_bin,
                "gap_bin": gap_bin,
                "density_bin": poi_desc["density_bin"],
                "centrality_bin": poi_desc["centrality_bin"],
                "urban_context_token": poi_desc["urban_context_token"],
            }

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

    session_tokens_df = pd.DataFrame(session_records)
    return session_tokens_df


# session_tokens_df = encode_session_spatial_tokens(
#     checkins_df=checkins_df,
#     poi_descriptor_df=poi_spatial_df,
#     pair_transition_df=pair_transition_df,
#     config=config,
# )

# session_tokens_df.head()
