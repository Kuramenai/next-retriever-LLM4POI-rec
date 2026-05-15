"""
LLM reranking module for next-POI prediction.

Takes top-k candidates from the learned reranker and uses an LLM
to make the final prediction based on contextual reasoning.

Supports multiple candidate ordering strategies for position bias ablation:
  - "reranker"     : keep learned reranker ordering (default)
  - "random"       : randomize order (baseline for measuring position bias)
  - "reverse"      : reverse reranker ordering (position bias diagnostic)
  - "by_distance"  : sort by distance from current location (nearest first)
  - "by_category"  : group by category, then by reranker score within group
  - "interleave"   : alternate high-ranked and low-ranked candidates

"""

from __future__ import annotations

from typing import Optional
from pathlib import Path
import pickle
import json
import re

import numpy as np
import pandas as pd
from termcolor import cprint

from tqdm import tqdm
from spatial_transition_retriever import TransitionIndex
from extract_poi_spatial_descriptors import SpatialEncodingConfig
from retrieve_decisions_states import DecisionStateRetrievalIndex, DecisionStateEncoder
from union_candidate_retriever import TrainedReranker, retrieve_and_rerank
from session_decision_state_table import build_current_decision_state

# ═══════════════════════════════════════════════════════════════════════════════
# Time helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _format_time(ts: pd.Timestamp) -> str:
    """Format timestamp as readable time string."""
    return ts.strftime("%I:%M %p").lstrip("0")


def _format_day(ts: pd.Timestamp) -> str:
    """Format timestamp as day of week."""
    return ts.strftime("%A")


def _time_of_day_label(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 14:
        return "lunchtime"
    elif 14 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "late night"


def _relative_direction(bearing_deg: float) -> str:
    """Convert bearing to human-readable direction."""
    dirs = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
    idx = int(((bearing_deg + 22.5) % 360) // 45)
    return dirs[idx]


def _bearing_between(lat1, lon1, lat2, lon2) -> float:
    """Compute bearing in degrees from point 1 to point 2."""
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    dlon = lon2_r - lon1_r
    y = np.sin(dlon) * np.cos(lat2_r)
    x = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_008.8
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    dlat, dlon = lat2_r - lat1_r, lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _format_distance(dist_m: float) -> str:
    """Human-readable distance."""
    if pd.isna(dist_m):
        return "unknown distance"
    if dist_m < 100:
        return f"{int(dist_m)}m"
    elif dist_m < 1000:
        return f"{int(round(dist_m, -1))}m"
    else:
        return f"{dist_m / 1000:.1f}km"


# ═══════════════════════════════════════════════════════════════════════════════
# Session narrative builder
# ═══════════════════════════════════════════════════════════════════════════════


def build_session_narrative(
    prefix_checkins_df: pd.DataFrame,
    poi_coord_map: dict[int, tuple[float, float]],
    config,
    *,
    recent_k: int = 4,
    max_early_summary: int = 3,
) -> dict:
    """
    Build a structured session narrative from check-in history.

    Returns a dict with:
      - "full_narrative": complete text narrative
      - "time_context": current time/day info
      - "recent_activity": last k check-ins as structured text
      - "early_summary": one-line summary of earlier activity
      - "current_poi": current POI info
      - "current_lat", "current_lon": current coordinates
    """
    df = prefix_checkins_df.copy()
    df[config.timestamp_col] = pd.to_datetime(df[config.timestamp_col])
    df = df.sort_values([config.timestamp_col, config.poi_id_col]).reset_index(drop=True)

    poi_lookup = {poi_id: coord for poi_id, coord in poi_coord_map.items()}

    # Current state
    last_row = df.iloc[-1]
    current_ts = last_row[config.timestamp_col]
    current_poi_id = last_row[config.poi_id_col]
    current_category = last_row.get(config.category_col, "Unknown")
    current_loc = poi_lookup.get(current_poi_id, {})
    current_lat = current_loc.get("lat", np.nan)
    current_lon = current_loc.get("lon", np.nan)

    # Time context
    hour = current_ts.hour
    time_context = (
        f"It is {_format_time(current_ts)} on a {_format_day(current_ts)} ({_time_of_day_label(hour)})."
    )

    # Build activity timeline
    n = len(df)
    split_point = max(0, n - recent_k)

    # Early activity summary
    early_summary = ""
    if split_point > 0:
        early_df = df.iloc[:split_point]
        early_categories = early_df[config.category_col].tolist()
        early_start = _format_time(early_df.iloc[0][config.timestamp_col])
        early_end = _format_time(early_df.iloc[-1][config.timestamp_col])

        # Deduplicate consecutive categories
        deduped = []
        for cat in early_categories:
            if not deduped or deduped[-1] != cat:
                deduped.append(str(cat))

        if len(deduped) <= max_early_summary:
            clean_cat_str = " -> ".join(deduped)
        else:
            clean_cat_str = (
                " -> ".join(deduped[:max_early_summary])
                + f" (and {len(deduped) - max_early_summary} more stops)"
            )
        early_summary = f"Earlier ({early_start} to {early_end}): {clean_cat_str}"

    # Recent activity (detailed)
    recent_df = df.iloc[split_point:]
    recent_lines = []
    prev_lat, prev_lon = np.nan, np.nan

    for _, row in recent_df.iterrows():
        ts = row[config.timestamp_col]
        poi_id = row[config.poi_id_col]
        category = row.get(config.category_col, "Unknown")
        loc = poi_lookup.get(poi_id, {})
        lat = loc.get("lat", np.nan)
        lon = loc.get("lon", np.nan)

        # Distance and direction from previous
        spatial_desc = ""
        if not np.isnan(prev_lat) and not np.isnan(lat):
            dist = _haversine_m(prev_lat, prev_lon, lat, lon)
            bearing = _bearing_between(prev_lat, prev_lon, lat, lon)
            direction = _relative_direction(bearing)
            spatial_desc = f", {_format_distance(dist)} {direction}"

        recent_lines.append(f"  {_format_time(ts)} — {category}{spatial_desc}")
        recent_lines[-1] = f"  {_format_time(ts)} - {category}{spatial_desc}"
        prev_lat, prev_lon = lat, lon

    recent_activity = "\n".join(recent_lines)

    # Current POI description
    current_poi_desc = f"The user is currently at: {current_category} (POI ID: {current_poi_id})"

    # Full narrative
    parts = [time_context, ""]
    if early_summary:
        parts.append(early_summary)
        parts.append("")
    parts.append("Recent activity:")
    parts.append(recent_activity)
    parts.append("")
    parts.append(current_poi_desc)

    return {
        "full_narrative": "\n".join(parts),
        "time_context": time_context,
        "recent_activity": recent_activity,
        "early_summary": early_summary,
        "current_poi": current_poi_desc,
        "current_lat": current_lat,
        "current_lon": current_lon,
        "current_category": str(current_category),
        "current_timestamp": current_ts,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Candidate formatting with ordering strategies
# ═══════════════════════════════════════════════════════════════════════════════

ORDERING_STRATEGIES = (
    "reranker",
    "random",
    "reverse",
    "by_distance",
    "by_category",
    "interleave",
)


def order_candidates(
    candidate_df: pd.DataFrame,
    ordering: str = "reranker",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Reorder candidates according to the specified strategy.

    Returns a copy with the new ordering (original index preserved for mapping back).
    """
    df = candidate_df.copy().reset_index(drop=True)
    df["_original_rank"] = range(1, len(df) + 1)

    if ordering == "reranker":
        return df

    elif ordering == "random":
        rng = np.random.default_rng(random_state)
        shuffled_idx = rng.permutation(len(df))
        return df.iloc[shuffled_idx].reset_index(drop=True)

    elif ordering == "reverse":
        return df.iloc[::-1].reset_index(drop=True)

    elif ordering == "by_distance":
        dist_col = "distance_m" if "distance_m" in df.columns else "distance_to_candidate_m"
        if dist_col in df.columns:
            return df.sort_values(dist_col, ascending=True, na_position="last").reset_index(drop=True)
        return df

    elif ordering == "by_category":
        cat_col = "next_category" if "next_category" in df.columns else None
        if cat_col and cat_col in df.columns:
            return df.sort_values([cat_col, "_original_rank"], ascending=[True, True]).reset_index(drop=True)
        return df

    elif ordering == "interleave":
        # Alternate: rank 1, rank 20, rank 2, rank 19, ...
        n = len(df)
        indices = []
        left, right = 0, n - 1
        while left <= right:
            indices.append(left)
            if left != right:
                indices.append(right)
            left += 1
            right -= 1
        return df.iloc[indices].reset_index(drop=True)

    else:
        raise ValueError(f"Unknown ordering strategy: {ordering!r}. Choose from: {ORDERING_STRATEGIES}")


def format_candidates_for_llm(
    candidate_df: pd.DataFrame,
    current_lat: float,
    current_lon: float,
    *,
    ordering: str = "reranker",
    random_state: int = 42,
    max_candidates: int = 20,
) -> tuple[str, pd.DataFrame]:
    """
    Format candidate POIs as a readable list for the LLM prompt.

    Returns:
      - formatted text string
      - ordered DataFrame (for mapping LLM response back to POI IDs)
    """
    ordered = order_candidates(
        candidate_df.head(max_candidates),
        ordering=ordering,
        random_state=random_state,
    )

    lines = []
    for idx, (_, row) in enumerate(ordered.iterrows(), start=1):
        poi_id = int(row["next_POIId"])
        category = row.get("next_category", "Unknown")
        if pd.isna(category) or category == "":
            category = "Unknown"

        # Distance and direction
        dist_m = row.get("distance_m", row.get("distance_to_candidate_m", np.nan))
        dist_str = _format_distance(dist_m) if not pd.isna(dist_m) else "unknown distance"

        line = f"{idx}. [ID: {poi_id}] {category} — {dist_str}"
        model_rank = int(row.get("_original_rank", idx))
        score = row.get("reranker_score", np.nan)
        score_text = f"; model_score={float(score):.4f}" if not pd.isna(score) else ""
        source_text = ""
        if "in_both_pools" in row and bool(row.get("in_both_pools", 0)):
            source_text = "; source=spatial+decision"
        elif "in_st_pool" in row and bool(row.get("in_st_pool", 0)):
            source_text = "; source=spatial"
        elif "in_ds_pool" in row and bool(row.get("in_ds_pool", 0)):
            source_text = "; source=decision"

        exact_text = ""
        if "st_from_exact_poi" in row:
            exact_text = f"; exact_transition={int(bool(row.get('st_from_exact_poi', 0)))}"

        support_parts = []
        for col, label in (
            ("st_transition_weight", "transition_weight"),
            ("st_context_score_max", "context_score"),
            ("ds_candidate_prob", "decision_prob"),
        ):
            value = row.get(col, np.nan)
            if not pd.isna(value):
                support_parts.append(f"{label}={float(value):.4f}")
        support_text = f"; evidence={', '.join(support_parts)}" if support_parts else ""
        line = (
            f"{idx}. candidate_number={idx}; poi_id={poi_id}; "
            f"category={category}; distance={dist_str}; "
            f"model_rank={model_rank}{score_text}{source_text}{exact_text}{support_text}"
        )
        lines.append(line)

    text = "Candidate next POIs:\n" + "\n".join(lines)
    return text, ordered


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt builder
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_SYSTEM_PROMPT = (
    "You are a location prediction system. Given a user's recent activity "
    "and a list of candidate Points of Interest (POIs), predict which ONE "
    "candidate POI the user will most likely visit next. "
    "Consider the time of day, the user's movement patterns, "
    "and the types of places they have been visiting. The candidates are "
    "already ordered by a trained ranking model, so use model_rank/model_score "
    "as a strong prior and only move away from the top-ranked candidates when "
    "the context strongly supports it."
)

DEFAULT_INSTRUCTION = (
    "Based on the user's activity pattern, current time, and location context, "
    "which candidate POI will the user most likely visit next?\n\n"
    "Respond with JSON only, using this exact schema: "
    '{"candidate_number": <one integer from the candidate list>}. '
    "Do not invent a POI and do not return explanatory text."
)


def build_reranking_prompt(
    prefix_checkins_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    poi_descriptor_df: pd.DataFrame,
    config,
    *,
    ordering: str = "reranker",
    recent_k: int = 4,
    max_candidates: int = 20,
    system_prompt: Optional[str] = None,
    instruction: Optional[str] = None,
    random_state: int = 42,
    include_system: bool = True,
) -> dict:
    """
    Build the complete LLM reranking prompt.

    Returns a dict with:
      - "system": system prompt string
      - "user": user message string
      - "ordered_candidates": DataFrame with the ordering used
      - "candidate_id_list": list of POI IDs in prompt order
      - "narrative": the session narrative dict
    """
    # Build narrative
    narrative = build_session_narrative(
        prefix_checkins_df=prefix_checkins_df,
        poi_descriptor_df=poi_descriptor_df,
        config=config,
        recent_k=recent_k,
    )

    # Format candidates
    candidates_text, ordered_df = format_candidates_for_llm(
        candidate_df=candidate_df,
        current_lat=narrative["current_lat"],
        current_lon=narrative["current_lon"],
        ordering=ordering,
        random_state=random_state,
        max_candidates=max_candidates,
    )

    # Assemble user message
    if instruction is None:
        instruction = DEFAULT_INSTRUCTION

    user_message = f"{narrative['full_narrative']}\n\n{candidates_text}\n\n{instruction}"

    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    return {
        "system": system_prompt if include_system else None,
        "user": user_message,
        "ordered_candidates": ordered_df,
        "candidate_id_list": [int(x) for x in ordered_df["next_POIId"]],
        "narrative": narrative,
    }


def llm_prompt_generator(
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
) -> tuple[list[dict], list[int]]:
    """Generate LLM prompts and gold next POI IDs for the given test checkins."""

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
    iterator = tqdm(groups, desc="Generate LLM prompts", unit="session") if show_progress else groups

    prompts = []
    gold_next_POIIds = []
    for session_id, session_df in iterator:
        session_df = session_df.sort_values([config.timestamp_col, config.poi_id_col]).reset_index(drop=True)

        if len(session_df) < min_checkins:
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

            prompt = build_reranking_prompt(
                prefix_checkins_df=prefix_df,
                candidate_df=candidate_pois,
                poi_descriptor_df=poi_descriptor_df,
                config=config,
                recent_k=recent_k,
            )
            prompts.append(prompt)
            gold_next_POIIds.append(_normalize(gold_poi_id))

        except Exception as e:
            cprint(f"Error generating prompt for session {session_id}: {e!r}", "red")
            gold_next_POIIds.append(None)

    return prompts, gold_next_POIIds


# ═══════════════════════════════════════════════════════════════════════════════
# Response parser
# ═══════════════════════════════════════════════════════════════════════════════


def parse_llm_response(
    response_text: str,
    ordered_candidates_df: pd.DataFrame,
) -> Optional[int]:
    """Backward-compatible wrapper returning only the predicted POI ID."""
    return parse_llm_response_with_metadata(response_text, ordered_candidates_df)["predicted_poi_id"]


def parse_llm_response_with_metadata(
    response_text: str,
    ordered_candidates_df: pd.DataFrame,
    *,
    prefer_position: bool = True,
) -> dict:
    """
    Extract the predicted POI ID and parser diagnostics from the LLM response.

    The current prompt asks for candidate_number, so list positions are preferred
    when the response is ambiguous. Explicit POI IDs are still accepted.
    """
    response = response_text.strip()
    valid_ids = set(int(x) for x in ordered_candidates_df["next_POIId"])

    def _empty(method: str = "parse_failed") -> dict:
        return {
            "predicted_poi_id": None,
            "selected_position": None,
            "parse_method": method,
            "raw_response": response_text,
        }

    def _from_position(pos: int, method: str) -> dict:
        if 1 <= pos <= len(ordered_candidates_df):
            return {
                "predicted_poi_id": int(ordered_candidates_df.iloc[pos - 1]["next_POIId"]),
                "selected_position": int(pos),
                "parse_method": method,
                "raw_response": response_text,
            }
        return _empty(method=f"{method}_out_of_range")

    def _from_poi_id(poi_id: int, method: str) -> dict:
        if poi_id in valid_ids:
            ordered_ids = [int(x) for x in ordered_candidates_df["next_POIId"]]
            return {
                "predicted_poi_id": int(poi_id),
                "selected_position": ordered_ids.index(int(poi_id)) + 1,
                "parse_method": method,
                "raw_response": response_text,
            }
        return _empty(method=f"{method}_invalid_id")

    # Strategy 1: JSON response from the recommended prompt.
    json_candidates = [response]
    json_match = re.search(r"\{.*\}", response, flags=re.DOTALL)
    if json_match:
        json_candidates.append(json_match.group(0))
    for candidate in json_candidates:
        try:
            parsed = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(parsed, dict):
            continue
        for key in ("candidate_number", "candidate", "choice", "option", "position", "rank"):
            if key in parsed:
                try:
                    return _from_position(int(parsed[key]), method=f"json_{key}")
                except (TypeError, ValueError):
                    pass
        for key in ("poi_id", "POIId", "next_POIId", "id"):
            if key in parsed:
                try:
                    return _from_poi_id(int(parsed[key]), method=f"json_{key}")
                except (TypeError, ValueError):
                    pass

    # Strategy 2: response is exactly one integer.
    try:
        as_int = int(response)
        if prefer_position and 1 <= as_int <= len(ordered_candidates_df):
            return _from_position(as_int, method="exact_position")
        if as_int in valid_ids:
            return _from_poi_id(as_int, method="exact_poi_id")
        if 1 <= as_int <= len(ordered_candidates_df):
            return _from_position(as_int, method="exact_position")
    except ValueError:
        pass

    # Strategy 3: explicit POI ID references.
    id_patterns = [
        r"(?:poi[_\s-]*id|next[_\s-]*poi[_\s-]*id|id)\s*[:=#-]?\s*(\d+)",
    ]
    for pattern in id_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return _from_poi_id(int(match.group(1)), method="explicit_poi_id")

    # Strategy 4: explicit list-position references.
    position_patterns = [
        r"(?:candidate[_\s-]*number|candidate|option|choice|number|rank|#)\s*[:=#-]?\s*(\d+)",
        r"^(\d+)\.",
        r"^(\d+)\b",
    ]
    for pattern in position_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return _from_position(int(match.group(1)), method="explicit_position")

    # Strategy 5: find any valid POI ID mentioned in the response.
    numbers = re.findall(r"\b(\d+)\b", response)
    for num_str in numbers:
        num = int(num_str)
        if num in valid_ids:
            return _from_poi_id(num, method="mentioned_poi_id")

    # Strategy 6: first number in response, treat as a list position.
    if numbers:
        return _from_position(int(numbers[0]), method="first_number_position")

    return _empty()


# ═══════════════════════════════════════════════════════════════════════════════
# Position bias diagnostic
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_position_bias(
    details_df: pd.DataFrame,
    ordering_col: str = "ordering",
) -> pd.DataFrame:
    """
    Analyze position bias across different ordering strategies.

    Expects details_df to have columns:
      - ordering: which strategy was used
      - gold_rank: rank of gold POI in the ordered list (None if not in list)
      - llm_predicted_position: which position the LLM selected

    Returns summary statistics per ordering strategy.
    """
    if ordering_col not in details_df.columns:
        print("No ordering column found. Run evaluation with multiple orderings first.")
        return pd.DataFrame()

    rows = []
    for ordering, group in details_df.groupby(ordering_col):
        valid = group[group["gold_next_POIId"].notna()]
        n = len(valid)
        if n == 0:
            continue

        hit1 = valid.get("hit@1", pd.Series(dtype=float))
        hit5 = valid.get("hit@5", pd.Series(dtype=float))

        # Position the LLM chose (not the gold rank)
        if "llm_selected_position" in valid.columns:
            pos = valid["llm_selected_position"].dropna()
            mean_selected_pos = pos.mean()
            selected_pos1_frac = (pos == 1).mean()
        else:
            mean_selected_pos = np.nan
            selected_pos1_frac = np.nan

        rows.append(
            {
                "ordering": ordering,
                "n_queries": n,
                "hit@1": hit1.mean() if len(hit1) > 0 else np.nan,
                "hit@5": hit5.mean() if len(hit5) > 0 else np.nan,
                "mean_selected_position": mean_selected_pos,
                "frac_selected_position_1": selected_pos1_frac,
            }
        )

    summary = pd.DataFrame(rows)
    if len(summary) > 0:
        print("\n" + "═" * 70)
        print("  POSITION BIAS ANALYSIS")
        print("═" * 70)
        print(summary.to_string(index=False, float_format="%.3f"))
        print("═" * 70)

    return summary


if __name__ == "__main__":
    city = "nyc"
    k_values = (1, 3, 5, 10, 20)

    scrip_dir = Path(__file__).resolve().parent.parent

    poi_descriptor_df = pd.read_csv(scrip_dir / f"artifacts/{city}/{city}_poi_descriptor.csv")

    with open(scrip_dir / f"artifacts/{city}/{city}_pair_lookup.pkl", "rb") as f:
        pair_lookup = pickle.load(f)
    with open(scrip_dir / f"artifacts/{city}/{city}_poi_coord_map.pkl", "rb") as f:
        poi_coord_map = pickle.load(f)
    lookup_df = pd.DataFrame.from_dict(pair_lookup, orient="index")
    lookup_df.index = pd.MultiIndex.from_tuples(lookup_df.index, names=["src_POIId", "dst_POIId"])  # fmt: skip
    coord_df = pd.DataFrame.from_dict(poi_coord_map, orient="index")

    config = SpatialEncodingConfig()
    sid_col = config.session_id_col
    ts_col = config.timestamp_col
    poi_id_col = config.poi_id_col

    test_checkins = pd.read_csv(scrip_dir / f"data/{city}/test_sample.csv")
    test_checkins = test_checkins.rename(columns={"pseudo_session_trajectory_id": sid_col})
    test_checkins[ts_col] = pd.to_datetime(test_checkins[ts_col], errors="coerce")

    # ── Generate prompts and gold next POI IDs ────────────────────

    source_tau = 300
    nearby_radius = 2000

    encoder_path = scrip_dir / f"artifacts/{city}/{city}_encoder.pkl"
    transition_index_path = scrip_dir / f"artifacts/{city}/{city}_transition_index.pkl"
    retrieval_index_path = scrip_dir / f"artifacts/{city}/{city}_retrieval_index.pkl"
    reranker_path = scrip_dir / f"artifacts/{city}/{city}_reranker.pkl"

    if (
        reranker_path.exists()
        and encoder_path.exists()
        and transition_index_path.exists()
        and retrieval_index_path.exists()
    ):
        with open(reranker_path, "rb") as f:
            reranker = pickle.load(f)
        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)
        with open(transition_index_path, "rb") as f:
            transition_index = pickle.load(f)
        with open(retrieval_index_path, "rb") as f:
            retrieval_index = pickle.load(f)
    else:
        raise FileNotFoundError(
            f"One or more required files not found: {reranker_path}, {encoder_path}, {transition_index_path}, {retrieval_index_path}"
        )

    recent_k = 4
    prompts, gold_next_POIIds = llm_prompt_generator(
        test_checkins_df=test_checkins,
        poi_descriptor_df=poi_descriptor_df,
        lookup_df=lookup_df,
        coord_df=coord_df,
        transition_index=transition_index,
        retrieval_index=retrieval_index,
        encoder=encoder,
        reranker=reranker,
        config=config,
        nearby_radius_m=nearby_radius,
        source_tau_m=source_tau,
        recent_k=recent_k,
        min_checkins=3,
    )

    out_dir = scrip_dir / f"artifacts/{city}"
    prompts_path = out_dir / f"{city}_llm_prompts.pkl"
    gold_next_POIIds_path = out_dir / f"{city}_gold_next_POIIds.pkl"

    with open(prompts_path, "wb") as f:
        pickle.dump(prompts, f)
    with open(gold_next_POIIds_path, "wb") as f:
        pickle.dump(gold_next_POIIds, f)

    cprint(f"Wrote prompts to {prompts_path.name}", "green")
    cprint(f"Wrote gold_next_POIIds to {gold_next_POIIds_path.name}", "green")
