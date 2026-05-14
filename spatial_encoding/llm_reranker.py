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

Usage
-----
    prompt = build_reranking_prompt(
        prefix_checkins_df=prefix_df,
        candidate_df=candidate_df,
        poi_descriptor_df=poi_descriptors,
        config=config,
        ordering="reranker",
    )

    # Call your LLM
    response = llm.generate(prompt)

    # Parse
    predicted_poi = parse_llm_response(response, candidate_df)
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# Time helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _format_time(ts: pd.Timestamp) -> str:
    """Format timestamp as readable time string."""
    return ts.strftime("%-I:%M %p")


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
    poi_descriptor_df: pd.DataFrame,
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

    # POI descriptor lookup
    poi_lookup = {}
    if poi_descriptor_df is not None:
        for _, row in poi_descriptor_df.drop_duplicates(subset=[config.poi_id_col]).iterrows():
            poi_lookup[row[config.poi_id_col]] = {
                "lat": float(row[config.lat_col]),
                "lon": float(row[config.lon_col]),
            }

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
            cat_str = " → ".join(deduped)
        else:
            cat_str = (
                " → ".join(deduped[:max_early_summary])
                + f" (and {len(deduped) - max_early_summary} more stops)"
            )

        early_summary = f"Earlier ({early_start}–{early_end}): {cat_str}"

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

        # Compute direction if we have candidate coordinates
        dir_str = ""
        # We can compute from distance_m if lat/lon are available in the transition index
        # For now just use distance

        line = f"{idx}. [ID: {poi_id}] {category} — {dist_str}"
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
    "and the types of places they have been visiting."
)

DEFAULT_INSTRUCTION = (
    "Based on the user's activity pattern, current time, and location context, "
    "which candidate POI will the user most likely visit next?\n\n"
    "Respond with ONLY the POI ID number, nothing else."
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


# ═══════════════════════════════════════════════════════════════════════════════
# Response parser
# ═══════════════════════════════════════════════════════════════════════════════


def parse_llm_response(
    response_text: str,
    ordered_candidates_df: pd.DataFrame,
) -> Optional[int]:
    """
    Extract the predicted POI ID from the LLM response.

    Tries multiple parsing strategies:
      1. Direct integer match against known candidate IDs
      2. List position reference ("1", "option 1", "#1")
      3. First integer found in response

    Returns the POI ID or None if parsing fails.
    """
    response = response_text.strip()
    valid_ids = set(int(x) for x in ordered_candidates_df["next_POIId"])

    # Strategy 1: response is exactly a valid POI ID
    try:
        as_int = int(response)
        if as_int in valid_ids:
            return as_int
    except ValueError:
        pass

    # Strategy 2: find any valid POI ID mentioned in the response
    import re

    numbers = re.findall(r"\b(\d+)\b", response)
    for num_str in numbers:
        num = int(num_str)
        if num in valid_ids:
            return num

    # Strategy 3: position reference (e.g., "candidate 1", "option 3")
    position_patterns = [
        r"(?:candidate|option|choice|number|#)\s*(\d+)",
        r"^(\d+)\.",
        r"^(\d+)\b",
    ]
    for pattern in position_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            pos = int(match.group(1))
            if 1 <= pos <= len(ordered_candidates_df):
                return int(ordered_candidates_df.iloc[pos - 1]["next_POIId"])

    # Strategy 4: first number in response, treat as position
    if numbers:
        pos = int(numbers[0])
        if 1 <= pos <= len(ordered_candidates_df):
            return int(ordered_candidates_df.iloc[pos - 1]["next_POIId"])

    return None


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
