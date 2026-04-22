import os
import sys
import re
import json
import pickle
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
from termcolor import cprint

# sys.path.append(os.path.abspath("/root/autodl-tmp/next-retriever-LLM4POI-rec"))
from spatial_encoding.retrieve_decisions_states import DecisionStateEncoder
from spatial_encoding.extract_poi_spatial_descriptors import SpatialEncodingConfig
from offline_mobility_prototype.prefix_feature_transformer import (
    FrozenModule1PrefixTransformer,
)
from end_to_end_pipeline import (
    EndToEndAssets,
    NextPOIEndToEndPipeline,
    Module1PrototypeRouter,
)

try:
    from vllm import LLM, SamplingParams  # type: ignore[import-not-found]
except ImportError as e:
    raise ImportError(
        "vllm is required for make_vllm_generate_fn. Install with: pip install vllm"
    ) from e


def _extract_json_object(text: str) -> str:
    """Strip optional ```json ... ``` fences and return a JSON object substring."""
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    # Raw JSON object/array
    return s


def parse_llm_selected_poi_id(llm_text: str, fallback_ids: list) -> Any:
    """
    Parse the LLM response for next-POI reranking (JSON schema from build_llm_reranking_prompt).

    Returns a POI id that appears in ``fallback_ids`` (same value / dtype as the matched
    fallback entry). If parsing fails or the model picks an invalid id, returns the first
    fallback id when available, else raises.
    """
    if not fallback_ids:
        raise ValueError("fallback_ids is empty; cannot select a next POI.")

    raw = _extract_json_object(llm_text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            "LLM output is not valid JSON. Ask for JSON-only output or use "
            "response_format=json_object when supported."
        ) from e

    if not isinstance(data, dict):
        raise ValueError("LLM JSON must be an object at the top level.")

    sel = data.get("selected_poi_id", None)
    if sel is None or (isinstance(sel, str) and not sel.strip()):
        ranking = data.get("ranking")
        if isinstance(ranking, list) and ranking:
            first = ranking[0]
            if isinstance(first, dict):
                sel = first.get("poi_id", sel)

    def _matches(candidate: Any, fid: Any) -> bool:
        if candidate == fid:
            return True
        if str(candidate).strip() == str(fid).strip():
            return True
        try:
            if int(candidate) == int(fid):
                return True
        except (TypeError, ValueError):
            pass
        return False

    if sel is not None:
        for fid in fallback_ids:
            if _matches(sel, fid):
                return fid

    return fallback_ids[0]


def stub_llm_generate_fn(system_prompt: str, user_prompt: str) -> str:
    """Deterministic JSON for dry-runs; parse_llm_selected_poi_id will fall back to first candidate."""
    return json.dumps(
        {
            "selected_poi_id": "",
            "ranking": [],
            "summary_reason": "stub_llm_generate_fn (no API call)",
        }
    )


def make_openai_chat_generate_fn(
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Callable[[str, str], str]:
    """
    Build ``llm_generate_fn(system_prompt, user_prompt) -> str`` using the OpenAI Chat Completions API.

    Uses ``OPENAI_API_KEY`` from the environment unless ``api_key`` is passed.
    Model defaults to ``OPENAI_MODEL`` env or ``gpt-4o-mini``.

    Request ``response_format={"type": "json_object"}`` so output matches the reranking schema.
    Requires: ``pip install openai`` (v1 SDK).
    """
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "openai package is required for make_openai_chat_generate_fn. "
            "Install with: pip install openai"
        ) from e

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OPENAI_API_KEY is not set and api_key was not provided to make_openai_chat_generate_fn."
        )

    resolved_model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    kwargs: dict[str, Any] = {"api_key": key}
    if base_url is not None:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)

    def generate(system_prompt: str, user_prompt: str) -> str:
        resp = client.chat.completions.create(
            model=resolved_model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        msg = resp.choices[0].message
        content = getattr(msg, "content", None) if msg is not None else None
        if not content:
            return ""
        return str(content)

    return generate


def make_vllm_generate_fn(
    *,
    model_path: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
    trust_remote_code: bool = True,
    tensor_parallel_size: int = 1,
    dtype: str = "auto",
    max_model_len: int = 32_768,
    gpu_memory_utilization: int = 0.95,
    **llm_kwargs: Any,
) -> Callable[[str, str], str]:
    """
    Build ``llm_generate_fn(system_prompt, user_prompt) -> str`` using vLLM in-process inference.

    Loads a HuggingFace-compatible model (e.g. Qwen3-8B) once and reuses it for every call.
    Prompts are formatted with the tokenizer's ``apply_chat_template`` when available
    (system + user messages), which matches typical Qwen chat usage.

    Parameters
    ----------
    model_path
        Directory with the HF model weights (``config.json``, tokenizer, etc.).
        Defaults to ``VLLM_MODEL_PATH`` env or ``/root/autodl-tmp/hf-models/Qwen3-8B``.
    temperature, max_tokens
        Passed to vLLM ``SamplingParams``.
    trust_remote_code
        Set True for many Qwen checkpoints.
    tensor_parallel_size
        vLLM tensor parallel degree.
    dtype
        ``"auto"`` or a torch dtype string accepted by vLLM.
    **llm_kwargs
        Extra keyword arguments forwarded to ``vllm.LLM(...)`` (e.g. ``gpu_memory_utilization``,
        ``max_model_len``, ``enforce_eager``).

    Requires
    --------
    ``pip install vllm`` and a CUDA-capable setup suitable for your model.

    Alternative (OpenAI-compatible server)
    ----------------------------------------
    If you prefer ``vllm serve /path/to/Qwen3-8B --host 0.0.0.0 --port 8000``,
    use ``make_openai_chat_generate_fn(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY", model="<served_model_id>")``
    instead (vLLM exposes an OpenAI-compatible HTTP API).
    """

    resolved_path = model_path

    engine_args: dict[str, Any] = {
        "model": resolved_path,
        "trust_remote_code": trust_remote_code,
        "tensor_parallel_size": int(tensor_parallel_size),
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
    }
    if dtype and str(dtype).lower() != "auto":
        engine_args["dtype"] = dtype
    engine_args.update(llm_kwargs)

    llm = LLM(**engine_args)
    tokenizer = llm.get_tokenizer()

    def generate(system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            prompt = f"{system_prompt}\n\n{user_prompt}"

        sampling = SamplingParams(
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        outputs = llm.generate([prompt], sampling)
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text.strip()

    return generate


def build_retrieval_caches(
    decision_state_case_base_df: pd.DataFrame,
    encoder: DecisionStateEncoder,
):
    """
    Precompute retrieval caches for fast inference-time retrieval.

    Returns:
      - case_vectors: (N, D) non-spatial block-weighted vectors
      - case_coords:  (N, 2) latitude/longitude for spatial kernel
    """
    encoder.fit(decision_state_case_base_df)
    case_vectors = encoder.transform(decision_state_case_base_df)
    case_coords = encoder.extract_coords(decision_state_case_base_df)
    return case_vectors, case_coords


if __name__ == "__main__":
    city = "nyc"
    scrip_dir = Path(__file__).resolve().parent.parent
    config = SpatialEncodingConfig()

    cprint("Loading artifacts", "yellow")
    with open(scrip_dir / f"artifacts/{city}/{city}_features.pkl", "rb") as f:
        features = pickle.load(f)

    poi_df = pd.read_csv(scrip_dir / f"artifacts/{city}/{city}_poi_meta.csv")
    poi_descriptor_df = pd.read_csv(
        scrip_dir / f"artifacts/{city}/{city}_poi_descriptor.csv"
    )
    pair_lookup_df = pd.read_csv(
        scrip_dir / f"artifacts/{city}/{city}_poi_pair_lookup_table.csv"
    )
    # Labeled historical decision states (must include next_POIId). Do not use the
    # pair-lookup table here — that CSV is src/dst pair geometry, not per-decision labels.
    decision_state_case_base_df = pd.read_csv(
        scrip_dir / f"artifacts/{city}/{city}_decision_state_table.csv"
    )

    cprint("Loading fitted GMM", "yellow")
    with open(scrip_dir / f"artifacts/{city}/{city}_gmm_cluster.pkl", "rb") as f:
        gmm = pickle.load(f)
    fitted_gmm = gmm["model"]

    cprint("Preparing online clustering", "yellow")
    frozen_module1_transformer = (
        FrozenModule1PrefixTransformer.from_feature_blocks_output(features)
    )

    cprint("Loading encoder, cases vectors and coords...", "yellow")
    # with open(scrip_dir / f"artifacts/{city}/{city}_case_vectors.pkl", "rb") as f:
    #     case_vectors = pickle.load(f)

    # with open(scrip_dir / f"artifacts/{city}/{city}_case_coords.pkl", "rb") as f:
    #     case_coords = pickle.load(f)

    # with open(scrip_dir / f"artifacts/{city}/{city}_case_encoder.pkl", "rb") as f:
    #     encoder = pickle.load(f)
    encoder = DecisionStateEncoder(config, recent_k=2)
    case_vectors, case_coords = build_retrieval_caches(
        decision_state_case_base_df, encoder
    )

    prototype_router = Module1PrototypeRouter(
        gmm_model=fitted_gmm,
        prefix_feature_transform_fn=frozen_module1_transformer.transform_prefix,  # see offline_mobility_prototype/prefix_feature_transformer.py
        feature_cols=frozen_module1_transformer.feature_cols,
        session_id_col=config.session_id_col,
    )

    assets = EndToEndAssets(
        config=config,
        poi_df=poi_df,
        poi_descriptor_df=poi_descriptor_df,
        pair_lookup_df=pair_lookup_df,
        decision_state_case_base_df=decision_state_case_base_df,
        decision_state_encoder=encoder,
        decision_state_case_vectors=case_vectors,
        decision_state_case_coords=case_coords,  # <-- add this for spatial kernel speed
        prototype_router=prototype_router,  # optional
        prototype_caption_map=None,  # optional
        pair_lookup_dict=None,  # optional (perf)
        poi_coord_map=None,  # optional (perf)
    )

    cprint("Launching VLLM", "yellow")
    model_path = "/root/autodl-tmp/hf-models/Qwen3-8B"
    # llm_generate_fn = make_openai_chat_generate_fn()  # needs OPENAI_API_KEY
    llm_generate_fn = make_vllm_generate_fn(model_path=model_path)
    llm_parse_fn = parse_llm_selected_poi_id

    pipeline = NextPOIEndToEndPipeline(
        assets=assets,
        llm_generate_fn=llm_generate_fn,
        llm_parse_fn=llm_parse_fn,
    )

    cprint("Inference", "yellow")
    test_sample_df = pd.read_csv(scrip_dir / f"data/{city}/test_sample.csv")

    # One session: slice a single SessionId, then predict last POI.
    # sid = test_sample_df[config.session_id_col].iloc[0]
    # one_session = test_sample_df[
    #     test_sample_df[config.session_id_col] == sid
    # ].copy()
    # out = pipeline.predict_from_full_test_session(one_session)

    # All sessions in test_sample_df (one next-step prediction per session).
    batch_results = pipeline.predict_batch_from_test_checkins(
        test_sample_df,
        min_checkins=2,
        include_details=False,
        show_progress=True,
    )
    evaluated = batch_results.loc[~batch_results["skipped"] & batch_results["error"].isna()]
    if len(evaluated) > 0:
        acc = float(evaluated["is_correct_at_1"].mean())
        cprint(f"Hit@1 over {len(evaluated)} sessions: {acc:.4f}", "green")
    out_path = scrip_dir / f"artifacts/{city}/{city}_batch_inference_results.csv"
    batch_results.to_csv(out_path, index=False)
    cprint(f"Wrote batch results to {out_path}", "green")
