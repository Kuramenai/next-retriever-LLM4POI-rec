from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from termcolor import cprint
from vllm import LLM, SamplingParams

from llm_reranker import parse_llm_response_with_metadata


def _normalize_poi_id(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return int(value) if float(value).is_integer() else float(value)
    value_str = str(value).strip()
    try:
        value_float = float(value_str)
        return int(value_float) if value_float.is_integer() else value_str
    except ValueError:
        return value_str


def _gold_rank_in_candidates(gold_poi_id, ordered_candidates: pd.DataFrame) -> int | None:
    gold_norm = _normalize_poi_id(gold_poi_id)
    for pos, candidate_id in enumerate(ordered_candidates["next_POIId"], start=1):
        if _normalize_poi_id(candidate_id) == gold_norm:
            return pos
    return None


if __name__ == "__main__":
    city = "nyc"
    script_dir = Path(__file__).resolve().parent.parent
    artifacts_dir = script_dir / f"artifacts/{city}"
    prompts_path = artifacts_dir / f"{city}_llm_prompts.pkl"
    gold_next_poi_ids_path = artifacts_dir / f"{city}_gold_next_POIIds.pkl"

    with open(prompts_path, "rb") as f:
        prompts = pickle.load(f)
    with open(gold_next_poi_ids_path, "rb") as f:
        gold_next_poi_ids = pickle.load(f)

    if len(prompts) != len(gold_next_poi_ids):
        raise ValueError(
            f"Prompt/gold length mismatch: {len(prompts)} prompts vs {len(gold_next_poi_ids)} gold labels."
        )
    prompts_with_candidate_number = sum("candidate_number" in str(prompt.get("user", "")) for prompt in prompts)
    prompts_with_model_rank = sum("model_rank" in str(prompt.get("user", "")) for prompt in prompts)
    cprint(
        f"Prompt format check: candidate_number={prompts_with_candidate_number}/{len(prompts)}, "
        f"model_rank={prompts_with_model_rank}/{len(prompts)}",
        "cyan",
    )
    if prompts_with_candidate_number < len(prompts) or prompts_with_model_rank < len(prompts):
        cprint(
            "Some prompts look like they were generated with an older prompt template. "
            "Regenerate the prompt pickle before trusting LLM metrics.",
            "yellow",
        )

    llm = LLM(model="/root/autodl-tmp/hf-models/Qwen3-8B", trust_remote_code=True)
    tokenizer = llm.get_tokenizer()

    # Use deterministic, short decoding for a forced-choice reranking task.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=32)

    messages = []
    for prompt in prompts:
        message = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]
        message = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        messages.append(message)

    outs = llm.generate(messages, sampling_params)
    texts: list[str] = []
    for output in outs:
        if not output.outputs:
            texts.append("")
        else:
            texts.append(output.outputs[0].text.strip())

    rows = []
    for i, (text, gold_next_poi_id, prompt) in enumerate(zip(texts, gold_next_poi_ids, prompts)):
        ordered_candidates = prompt["ordered_candidates"]
        parse = parse_llm_response_with_metadata(text, ordered_candidates, prefer_position=True)
        predicted_next_poi_id = parse["predicted_poi_id"]

        gold_norm = _normalize_poi_id(gold_next_poi_id)
        pred_norm = _normalize_poi_id(predicted_next_poi_id)
        gold_rank = _gold_rank_in_candidates(gold_next_poi_id, ordered_candidates)
        ranker_top1 = (
            _normalize_poi_id(ordered_candidates.iloc[0]["next_POIId"])
            if len(ordered_candidates) > 0
            else None
        )

        rows.append(
            {
                "query_index": i,
                "gold_next_POIId": gold_next_poi_id,
                "predicted_next_POIId": predicted_next_poi_id,
                "selected_position": parse["selected_position"],
                "parse_method": parse["parse_method"],
                "is_parse_failure": predicted_next_poi_id is None,
                "is_correct_at_1": pred_norm == gold_norm,
                "ranker_top1_POIId": ranker_top1,
                "ranker_top1_hit": ranker_top1 == gold_norm,
                "gold_rank_in_prompt": gold_rank,
                "gold_in_prompt": gold_rank is not None,
                "raw_response": text,
            }
        )

    details = pd.DataFrame(rows)
    hit1 = float(details["is_correct_at_1"].mean()) if len(details) else np.nan
    ranker_hit1 = float(details["ranker_top1_hit"].mean()) if len(details) else np.nan
    prompt_oracle_hit = float(details["gold_in_prompt"].mean()) if len(details) else np.nan
    parse_failure_rate = float(details["is_parse_failure"].mean()) if len(details) else np.nan
    selected_positions = pd.to_numeric(details["selected_position"], errors="coerce")
    max_prompt_candidates = max((len(prompt["ordered_candidates"]) for prompt in prompts), default=0)

    cprint(f"LLM Hit@1: {hit1:.4f}", "green")
    cprint(f"Ranker top-1 hit on same prompts: {ranker_hit1:.4f}", "cyan")
    cprint(f"Prompt oracle hit@{max_prompt_candidates}: {prompt_oracle_hit:.4f}", "cyan")
    cprint(f"Parse failure rate: {parse_failure_rate:.4f}", "cyan")
    cprint(f"Mean selected position: {selected_positions.mean():.2f}", "cyan")
    cprint(f"Selected position 1 fraction: {(selected_positions == 1).mean():.4f}", "cyan")

    cprint("Guardrail hit@1 if accepting the LLM only within a rank cap:", "cyan")
    for cap in (1, 2, 3, 5, 10, max_prompt_candidates):
        if cap <= 0:
            continue
        use_llm = selected_positions <= cap
        guarded_hit = np.where(use_llm, details["is_correct_at_1"], details["ranker_top1_hit"])
        cprint(
            f"  accept selected_position <= {cap:<2d}: {float(np.mean(guarded_hit)):.4f} "
            f"(accepted {float(use_llm.mean()):.4f})",
            "cyan",
        )

    pos_summary = (
        details.dropna(subset=["selected_position"])
        .groupby("selected_position")["is_correct_at_1"]
        .agg(["count", "mean"])
        .head(10)
    )
    cprint("Hit rate by selected position, first 10 positions:", "cyan")
    cprint(pos_summary.to_string(float_format="%.4f"), "cyan")

    parse_counts = details["parse_method"].value_counts(dropna=False).head(10)
    cprint("Top parse methods:", "cyan")
    for method, count in parse_counts.items():
        cprint(f"  {method}: {count}", "cyan")

    responses_path = artifacts_dir / f"{city}_llm_responses.pkl"
    details_path = artifacts_dir / f"{city}_llm_details.csv"
    with open(responses_path, "wb") as f:
        pickle.dump(texts, f)
    details.to_csv(details_path, index=False)
    cprint(f"Wrote llm responses to {responses_path}", "green")
    cprint(f"Wrote llm details to {details_path}", "green")
