import pickle
from pathlib import Path
from vllm import LLM, SamplingParams
from termcolor import cprint
from llm_reranker import parse_llm_response
import pandas as pd


if __name__ == "__main__":
    city = "nyc"
    script_dir = Path(__file__).resolve().parent.parent
    prompts_path = script_dir / f"artifacts/{city}/{city}_llm_prompts.pkl"
    gold_next_POIIds_path = script_dir / f"artifacts/{city}/{city}_gold_next_POIIds.pkl"
    with open(prompts_path, "rb") as f:
        prompts = pickle.load(f)
    with open(gold_next_POIIds_path, "rb") as f:
        gold_next_POIIds = pickle.load(f)

    llm = LLM(model="/root/autodl-tmp/hf-models/Qwen3-8B", trust_remote_code=True)
    tokenizer = llm.get_tokenizer()
    temperature = 0.2
    max_tokens = 1024
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
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
    for o in outs:
        if not o.outputs:
            texts.append("")
        else:
            texts.append(o.outputs[0].text.strip())

    metrics = []
    for text, gold_next_POIId, prompt in zip(texts, gold_next_POIIds, prompts):
        predicted_next_POIId = parse_llm_response(text, prompt["ordered_candidates"])
        metrics.append(
            {
                "gold_next_POIId": gold_next_POIId,
                "predicted_next_POIId": predicted_next_POIId,
                "is_correct_at_1": predicted_next_POIId == gold_next_POIId,
            }
        )

    cprint(f"Hit@1: {pd.DataFrame(metrics)['is_correct_at_1'].mean():.4f}", "green")

    with open(script_dir / f"artifacts/{city}/{city}_llm_responses.pkl", "wb") as f:
        pickle.dump(texts, f)
    cprint(
        f"Wrote llm responses to {script_dir / f'artifacts/{city}/{city}_llm_responses.pkl'.name}", "green"
    )
