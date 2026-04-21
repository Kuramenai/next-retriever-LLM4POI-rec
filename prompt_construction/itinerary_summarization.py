from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _safe_str(x) -> str:
    if pd.isna(x):
        return "unknown"
    return str(x)


def _format_timestamp(ts) -> str:
    if pd.isna(ts):
        return "unknown time"
    ts = pd.to_datetime(ts)
    return ts.strftime("%b %d %H:%M")


def _resolve_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _describe_exploration(
    prefix_repeat_ratio: float, prefix_unique_poi_count: int
) -> str:
    if pd.isna(prefix_repeat_ratio):
        return "The observed prefix does not clearly indicate whether the user is repeating stops or exploring new ones."

    if prefix_repeat_ratio <= 0.10:
        return (
            "The observed prefix is mostly exploratory, with little repetition so far."
        )
    if prefix_repeat_ratio <= 0.30:
        return "The observed prefix is moderately exploratory, with some repetition but mostly new stops."
    return "The observed prefix shows noticeable repetition, suggesting a more routine or backtracking pattern."


def _describe_area_context(row: pd.Series) -> str:
    density = row.get("curr_density_count", np.nan)
    degree = row.get("curr_node_degree", np.nan)

    density_txt = None
    degree_txt = None

    if pd.notna(density):
        if density >= 100:
            density_txt = "a very dense urban zone"
        elif density >= 30:
            density_txt = "a moderately dense urban zone"
        else:
            density_txt = "a relatively sparse local area"

    if pd.notna(degree):
        if degree >= 4:
            degree_txt = "with high local street connectivity"
        elif degree >= 2:
            degree_txt = "with moderate local street connectivity"
        else:
            degree_txt = "with limited local street connectivity"

    if density_txt and degree_txt:
        return f"The current stop is located in {density_txt} {degree_txt}."
    if density_txt:
        return f"The current stop is located in {density_txt}."
    if degree_txt:
        return f"The current stop is located in an area {degree_txt}."
    return "The local spatial context of the current stop is not strongly characterized by the available density/connectivity features."


def _describe_recent_movement(row: pd.Series) -> str:
    d = row.get("prev1_distance_m", np.nan)
    g = row.get("prev1_gap_s", np.nan)
    dir_bin = row.get("prev1_direction_bin", None)

    if pd.isna(d) or pd.isna(g):
        # fall back to bins if raw values unavailable
        d_bin = row.get("prev1_distance_bin", "unknown")
        g_bin = row.get("prev1_gap_bin", "unknown")
        if d_bin == "BOS":
            return "No prior movement is available because the current prefix contains only one observed stop."
        return f"The most recent transition was in distance bin {d_bin} with time-gap bin {g_bin}."

    minutes = max(float(g) / 60.0, 1e-6)
    pace = float(d) / minutes  # meters per minute

    if d < 250 and g < 1800:
        movement_txt = "The most recent transition was compact and local."
    elif d < 1000 and g < 3600:
        movement_txt = ("The most recent transition was moderate in both distance and time.")  # fmt: skip
    elif d >= 1000 and g < 3600:
        movement_txt = "The most recent transition covered a relatively long distance in a limited amount of time."
    else:
        movement_txt = "The most recent transition suggests a longer move or a slower-paced gap between stops."

    if pd.notna(dir_bin) and dir_bin not in [None, "unknown", "BOS"]:
        movement_txt += f" Its dominant direction bin was {dir_bin}."

    movement_txt += (f" Approximate recent movement pace was {pace:.1f} meters per minute.")  # fmt: skip
    return movement_txt


def _describe_time_context(row: pd.Series) -> str:
    tbin = row.get("current_time_bin", None)
    elapsed = row.get("prefix_elapsed_min", np.nan)

    if pd.isna(elapsed):
        return f"The current decision point occurs during the {tbin or 'unknown part of day'}."

    hours = float(elapsed) / 60.0
    return (
        f"The current decision point occurs during the {tbin or 'unknown part of day'}, "
        f"after about {hours:.1f} hours of observed activity in the current prefix."
    )


def _summarize_candidate_support(
    candidate_pois_df: pd.DataFrame, poi_meta_df: pd.DataFrame
) -> str:
    if len(candidate_pois_df) == 0:
        return "No candidate next POIs were retrieved from historical decision states."

    poi_id_col = _resolve_col(
        poi_meta_df, ["POIId", "PoiId", "VenueId", "venue_id", "poi_id"]
    )
    name_col = _resolve_col(
        poi_meta_df, ["PoiName", "POIName", "VenueName", "venue_name", "name"]
    )
    cat_col = _resolve_col(
        poi_meta_df, ["CategoryName", "category", "PoiCategory", "VenueCategory"]
    )

    if poi_id_col is None:
        raise ValueError("poi_meta_df must contain a POI id column.")

    meta = poi_meta_df.drop_duplicates(subset=[poi_id_col], keep="first").copy()

    merged = candidate_pois_df.merge(
        meta[[c for c in [poi_id_col, name_col, cat_col] if c is not None]],
        left_on="next_POIId",
        right_on=poi_id_col,
        how="left",
    )

    lines = []
    for i, r in merged.head(5).iterrows():
        nm = (
            _safe_str(r[name_col])
            if name_col is not None and name_col in merged.columns
            else _safe_str(r["next_POIId"])
        )
        ct = (
            _safe_str(r[cat_col])
            if cat_col is not None and cat_col in merged.columns
            else "unknown category"
        )
        prob = float(r.get("candidate_prob", np.nan))
        support = (
            int(r.get("support_case_count", 0))
            if pd.notna(r.get("support_case_count", np.nan))
            else 0
        )
        lines.append(
            f"{len(lines) + 1}. {nm} ({ct}) — estimated support probability {prob:.3f}, supported by {support} retrieved cases."
        )

    return "\n".join(lines)


def build_prompt_ready_evidence_block(
    partial_session_df: pd.DataFrame,
    current_state_df: pd.DataFrame,
    retrieved_cases_df: pd.DataFrame,
    candidate_pois_df: pd.DataFrame,
    poi_meta_df: pd.DataFrame,
    config,
    *,
    prototype_caption: Optional[str] = None,
    max_recent_stops: int = 4,
    max_evidence_cases: int = 3,
    max_candidates: int = 5,
) -> str:
    """
    Convert the current decision state + retrieved evidence into a prompt-ready
    natural-language block for LLM reranking.
    """

    if len(current_state_df) != 1:
        raise ValueError("current_state_df must contain exactly one row.")

    q = current_state_df.iloc[0]

    poi_id_col = _resolve_col(poi_meta_df, ["POIId", "PoiId", "VenueId", "venue_id", "poi_id"])  # fmt:skip
    name_col = _resolve_col(poi_meta_df, ["PoiName", "POIName", "VenueName", "venue_name", "name"])  # fmt: skip
    addr_col = _resolve_col(poi_meta_df, ["Address", "address", "formatted_address", "PoiAddress"])  # fmt: skip
    cat_col = _resolve_col(poi_meta_df, ["CategoryName", "category", "PoiCategory", "VenueCategory"])  # fmt: skip

    if poi_id_col is None:
        raise ValueError("poi_meta_df must contain a POI id column.")

    meta = poi_meta_df.drop_duplicates(subset=[poi_id_col], keep="first").set_index(poi_id_col)  # fmt: skip

    # ------------------------------------------------------------
    # 1) Current mobility summary
    # ------------------------------------------------------------
    summary_parts = [
        _describe_time_context(q),
        _describe_recent_movement(q),
        _describe_exploration(
            prefix_repeat_ratio=q.get("prefix_repeat_ratio", np.nan),
            prefix_unique_poi_count=q.get("prefix_unique_poi_count", np.nan),
        ),
        _describe_area_context(q),
    ]
    if prototype_caption:
        summary_parts.insert(
            0,
            f"The routed behavioral prototype suggests the following broad mobility pattern: {prototype_caption}",
        )

    mobility_summary = " ".join(summary_parts)

    # ------------------------------------------------------------
    # 2) Observed itinerary
    # ------------------------------------------------------------
    itinerary_lines = []
    obs_df = partial_session_df.copy()
    obs_df[config.timestamp_col] = pd.to_datetime(obs_df[config.timestamp_col], errors="coerce")  # fmt: skip
    obs_df = (
        obs_df.sort_values(config.timestamp_col)
        .tail(max_recent_stops)
        .reset_index(drop=True)
    )

    for idx, row in obs_df.iterrows():
        poi_id = row[config.poi_id_col]
        ts_txt = _format_timestamp(row[config.timestamp_col])

        if poi_id in meta.index:
            m = meta.loc[poi_id]
            nm = _safe_str(m[name_col]) if name_col is not None else _safe_str(poi_id)
            ct = (
                _safe_str(m[cat_col])
                if cat_col is not None
                else _safe_str(row.get(getattr(config, "category_col", ""), "unknown"))
            )
            ad = _safe_str(m[addr_col]) if addr_col is not None else "unknown address"
        else:
            nm = _safe_str(poi_id)
            ct = _safe_str(row.get(getattr(config, "category_col", ""), "unknown"))
            ad = "unknown address"

        itinerary_lines.append(f"Stop {idx + 1}: {nm}, a {ct}, at {ad} ({ts_txt})")

    itinerary_block = (
        "\n".join(itinerary_lines)
        if itinerary_lines
        else "No observed stops are available."
    )

    # ------------------------------------------------------------
    # 3) Historical evidence
    # ------------------------------------------------------------
    evidence_lines = []
    if len(retrieved_cases_df) > 0:
        for _, r in retrieved_cases_df.head(max_evidence_cases).iterrows():
            next_poi = r.get("next_POIId", None)
            score = r.get("retrieval_score", np.nan)

            if pd.notna(next_poi) and next_poi in meta.index:
                m = meta.loc[next_poi]
                nm = (
                    _safe_str(m[name_col])
                    if name_col is not None
                    else _safe_str(next_poi)
                )
                ct = (
                    _safe_str(m[cat_col]) if cat_col is not None else "unknown category"
                )
                ad = (
                    _safe_str(m[addr_col])
                    if addr_col is not None
                    else "unknown address"
                )
                evidence_lines.append(
                    f"- A closely matched historical decision state continued to {nm} ({ct}) at {ad} with retrieval score {float(score):.4f}."
                )
            else:
                evidence_lines.append(
                    f"- A closely matched historical decision state continued to POI {next_poi} with retrieval score {float(score):.4f}."
                )
    else:
        evidence_lines.append(
            "- No closely matched historical decision states were retrieved."
        )

    historical_block = "\n".join(evidence_lines)

    # ------------------------------------------------------------
    # 4) Candidate next POIs
    # ------------------------------------------------------------
    candidate_block = _summarize_candidate_support(
        candidate_pois_df.head(max_candidates),
        poi_meta_df=poi_meta_df,
    )

    # ------------------------------------------------------------
    # 5) Final prompt-ready evidence block
    # ------------------------------------------------------------
    text = f"""Current mobility summary:
{mobility_summary}

Observed itinerary so far:
{itinerary_block}

Historical decision-state evidence:
{historical_block}

Candidate next POIs inferred from retrieved historical cases:
{candidate_block}
"""
    return text.strip()
