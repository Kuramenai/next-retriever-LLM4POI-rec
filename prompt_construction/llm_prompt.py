from __future__ import annotations

from typing import Optional

import pandas as pd


def _resolve_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_str(x) -> str:
    if pd.isna(x):
        return "unknown"
    return str(x)


def _format_candidate_list_for_prompt(
    candidate_pois_df: pd.DataFrame,
    poi_meta_df: pd.DataFrame,
    *,
    max_candidates: int = 10,
) -> str:
    if len(candidate_pois_df) == 0:
        return "No candidate POIs are available."

    poi_id_col = _resolve_col(
        poi_meta_df, ["POIId", "PoiId", "VenueId", "venue_id", "poi_id"]
    )
    name_col = _resolve_col(
        poi_meta_df, ["PoiName", "POIName", "VenueName", "venue_name", "name"]
    )
    addr_col = _resolve_col(
        poi_meta_df, ["Address", "address", "formatted_address", "PoiAddress"]
    )
    cat_col = _resolve_col(
        poi_meta_df, ["PoiCategoryName", "category", "PoiCategory", "VenueCategory"]
    )

    if poi_id_col is None:
        raise ValueError("poi_meta_df must contain a POI id column.")

    meta = poi_meta_df.drop_duplicates(subset=[poi_id_col], keep="first").set_index(
        poi_id_col
    )

    lines = []
    for rank, (_, row) in enumerate(
        candidate_pois_df.head(max_candidates).iterrows(), start=1
    ):
        poi_id = row["next_POIId"]

        if poi_id in meta.index:
            m = meta.loc[poi_id]
            name = _safe_str(m[name_col]) if name_col is not None else _safe_str(poi_id)
            addr = _safe_str(m[addr_col]) if addr_col is not None else "unknown address"
            cat = (
                _safe_str(m[cat_col])
                if cat_col is not None
                else _safe_str(row.get("next_category", "unknown"))
            )
        else:
            name = _safe_str(poi_id)
            addr = "unknown address"
            cat = _safe_str(row.get("next_category", "unknown"))

        prob = row.get("candidate_prob", None)
        support = row.get("support_case_count", None)

        line = (
            f"Candidate {rank} | "
            f"POI_ID={poi_id} | "
            f"Name={name} | "
            f"Category={cat} | "
            f"Address={addr}"
        )

        if prob is not None and pd.notna(prob):
            line += f" | RetrievedSupportProb={float(prob):.4f}"
        if support is not None and pd.notna(support):
            line += f" | RetrievedSupportCases={int(support)}"

        lines.append(line)

    return "\n".join(lines)


def build_llm_reranking_prompt(
    evidence_block: str,
    candidate_pois_df: pd.DataFrame,
    poi_meta_df: pd.DataFrame,
    *,
    max_candidates: int = 10,
    top_k_return: int = 5,
) -> dict:
    """
    Build a grounded LLM reranking prompt from:
      - natural-language evidence block
      - candidate POI set
    """

    candidate_block = _format_candidate_list_for_prompt(
        candidate_pois_df=candidate_pois_df,
        poi_meta_df=poi_meta_df,
        max_candidates=max_candidates,
    )

    system_prompt = f"""You are a next-POI reranking assistant for urban mobility recommendation.

Your task is to rank the provided candidate POIs for the user's next stop.

Rules:
1. You must rank ONLY the provided candidate POIs.
2. Use the evidence block and retrieved-support signals as grounded evidence.
3. Prefer candidates that are coherent with:
   - the observed itinerary so far
   - the current time context
   - recent movement pattern
   - local spatial context
   - retrieved historical decision-state evidence
4. Do NOT invent facts such as budget, mood, or preferences unless directly supported by the evidence.
5. If two candidates are close, prefer the one with stronger retrieved historical support.
6. Return valid JSON only.

Return JSON with this schema:
{{
  "selected_poi_id": "<best candidate POI id>",
  "ranking": [
    {{
      "rank": 1,
      "poi_id": "<candidate POI id>",
      "justification": "<short grounded explanation>"
    }}
  ],
  "summary_reason": "<1-2 sentence overall explanation>"
}}

Return exactly the top {top_k_return} ranked candidates in the ranking list.
"""

    user_prompt = f"""Evidence block:
{evidence_block}

Candidate next POIs:
{candidate_block}

Please rank the candidates and select the best next POI.
"""

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


if __name__ == "__main__":
    pass

# evidence_block = build_prompt_ready_evidence_block(
#     partial_session_df=current_prefix_df,
#     current_state_df=current_state_df,
#     retrieved_cases_df=retrieved_cases_df,
#     candidate_pois_df=candidate_pois_df,
#     poi_meta_df=poi_df,
#     config=config,
#     prototype_caption=None,   # optional later
# )

# prompt_payload = build_llm_reranking_prompt(
#     evidence_block=evidence_block,
#     candidate_pois_df=candidate_pois_df,
#     poi_meta_df=poi_df,
#     max_candidates=10,
#     top_k_return=5,
# )

# system_prompt = prompt_payload["system_prompt"]
# user_prompt = prompt_payload["user_prompt"]
