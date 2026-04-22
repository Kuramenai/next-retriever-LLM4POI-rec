from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

import pickle
from pathlib import Path
import pandas as pd
import numpy as np

from offline_mobility_prototype.prefix_feature_transformer import (
    FrozenModule1PrefixTransformer,
)

from spatial_encoding.extract_poi_spatial_descriptors import SpatialEncodingConfig
from spatial_encoding.retrieve_decisions_states import DecisionStateEncoder
from spatial_encoding.session_decision_state_table import build_current_decision_state
from spatial_encoding.retrieve_decisions_states import retrieve_similar_decision_states
from spatial_encoding.retrieve_candidates_pois import (
    aggregate_candidate_pois_from_retrieved_cases,
)

from prompt_construction.itinerary_summarization import (
    build_prompt_ready_evidence_block,
)
from prompt_construction.llm_prompt import build_llm_reranking_prompt

from spatial_encoding.retrieve_decisions_states import RetrievalBlockWeights
import __main__

__main__.RetrievalBlockWeights = RetrievalBlockWeights


# ---------------------------------------------------------------------------
# LLM: generate + parse (wired into NextPOIEndToEndPipeline)
# ---------------------------------------------------------------------------


@dataclass
class Module1PrototypeRouter:
    """
    Inference-time router for Module 1 GMM prototypes.

    Parameters
    ----------
    gmm_model :
        Fitted sklearn GaussianMixture from Module 1 training.
    prefix_feature_transform_fn :
        Callable that transforms a single prefix/session DataFrame into the
        SAME feature schema used to fit the GMM. It must use frozen artifacts
        from training (e.g., stored TF-IDF / compression objects), not refit.
    feature_cols :
        Ordered list of columns expected by the GMM.
    session_id_col :
        Session id column name.
    min_prefix_len :
        Minimum observed check-ins required before routing is considered reliable.
    default_top_m :
        Number of top prototypes to expose by default.
    """

    gmm_model: object
    prefix_feature_transform_fn: Callable[[pd.DataFrame], pd.DataFrame]
    feature_cols: list[str]
    session_id_col: str = "SessionId"
    min_prefix_len: int = 3
    default_top_m: int = 3

    def predict_prefix(
        self,
        prefix_df: pd.DataFrame,
        *,
        top_m: Optional[int] = None,
    ) -> dict:
        top_m = self.default_top_m if top_m is None else int(top_m)

        if not isinstance(prefix_df, pd.DataFrame):
            raise TypeError("prefix_df must be a pandas DataFrame.")
        if len(prefix_df) == 0:
            raise ValueError("prefix_df is empty.")

        n_prefix = len(prefix_df)

        # Cold-start / too-short prefix: bypass routing
        if n_prefix < self.min_prefix_len:
            rec = {
                "routing_applied": False,
                "n_prefix_checkins": int(n_prefix),
                "prototype_id": np.nan,
                "prototype_prob": np.nan,
                "top3_mass": np.nan,
                "posterior_entropy": np.nan,
            }
            for r in range(1, max(top_m, 3) + 1):
                rec[f"top{r}_prototype_id"] = np.nan
                rec[f"top{r}_prototype_prob"] = np.nan
            return rec

        feat_df = self.prefix_feature_transform_fn(prefix_df)

        if not isinstance(feat_df, pd.DataFrame):
            raise TypeError("prefix_feature_transform_fn must return a DataFrame.")
        if len(feat_df) != 1:
            raise ValueError(
                "prefix_feature_transform_fn must return exactly one row for one prefix."
            )

        # Align to the exact training schema
        x_df = feat_df.reindex(columns=self.feature_cols, fill_value=0.0)

        # Guard against non-numeric surprise columns
        try:
            x = x_df.to_numpy(dtype=float)
        except Exception as e:
            raise ValueError(
                "Failed to convert prefix features to numeric array. "
                "Check that all Module 1 feature columns are numeric and aligned."
            ) from e

        # Posterior over GMM components
        posterior = self.gmm_model.predict_proba(x)[0]  # shape: (K,)
        order = np.argsort(posterior)[::-1]

        k = min(top_m, len(order))
        top_ids = order[:k]
        top_probs = posterior[top_ids]

        entropy = -float(np.sum(posterior * np.log(np.clip(posterior, 1e-12, 1.0))))
        top_mass = float(np.sum(top_probs))

        rec = {
            "routing_applied": True,
            "n_prefix_checkins": int(n_prefix),
            "prototype_id": int(top_ids[0]),
            "prototype_prob": float(top_probs[0]),
            f"top{k}_mass": top_mass,  # e.g. top3_mass if k=3
            "top3_mass": float(np.sum(posterior[order[: min(3, len(order))]])),
            "posterior_entropy": entropy,
        }

        # Always expose at least top-3 slots for downstream convenience
        expose_n = max(3, top_m)
        for r in range(1, expose_n + 1):
            if r <= k:
                rec[f"top{r}_prototype_id"] = int(top_ids[r - 1])
                rec[f"top{r}_prototype_prob"] = float(top_probs[r - 1])
            else:
                rec[f"top{r}_prototype_id"] = np.nan
                rec[f"top{r}_prototype_prob"] = np.nan

        return rec


@dataclass
class EndToEndAssets:
    config: object
    poi_df: pd.DataFrame
    poi_descriptor_df: pd.DataFrame
    pair_lookup_df: pd.DataFrame

    # Module 1 / retrieval assets
    decision_state_case_base_df: pd.DataFrame
    decision_state_encoder: object
    decision_state_case_vectors: object
    decision_state_case_coords: Optional[object] = None
    decision_state_retrieval_index: Optional[object] = None

    # Optional Module 1 router
    prototype_router: Optional[object] = None
    prototype_caption_map: Optional[dict] = None

    # Optional prebuilt online lookup helpers
    pair_lookup_dict: Optional[dict] = None
    poi_coord_map: Optional[dict] = None

    recent_k: int = 2
    top_k_return: int = 1
    top_k_retrieved_cases: int = 20
    top_k_candidates: int = 10
    max_recent_stops: int = 4
    max_evidence_cases: int = 3
    max_candidates: int = 5
    prototype_union_k: int = 1


class NextPOIEndToEndPipeline:
    def __init__(
        self,
        assets: EndToEndAssets,
        llm_generate_fn: Callable[[str, str], str],
        llm_parse_fn: Callable[[str, list], str],
    ):
        self.assets = assets
        self.llm_generate_fn = llm_generate_fn
        self.llm_parse_fn = llm_parse_fn

    # ------------------------------------------------------------
    # 1) Query construction from one full test session
    # ------------------------------------------------------------
    def build_test_query_from_full_session(
        self,
        full_session_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        config = self.assets.config
        work = full_session_df.copy()
        work[config.timestamp_col] = pd.to_datetime(
            work[config.timestamp_col], errors="coerce"
        )
        work = work.sort_values(config.timestamp_col).reset_index(drop=True)

        if config.session_id_col in work.columns:
            n_sessions = work[config.session_id_col].dropna().nunique()
            if n_sessions > 1:
                raise ValueError(
                    f"full_session_df must contain one session only; found {n_sessions} "
                    f"distinct {config.session_id_col!r}. Use "
                    f"predict_batch_from_test_checkins(test_checkins_df) for the full test CSV."
                )

        if len(work) < 2:
            raise ValueError("A test session must contain at least 2 check-ins.")

        prefix_df = work.iloc[:-1].copy().reset_index(drop=True)
        gold_next = work.iloc[-1].copy()
        return prefix_df, gold_next

    # ------------------------------------------------------------
    # 2) Module 1 routing on prefix
    # ------------------------------------------------------------
    def infer_prototype_signals(
        self,
        prefix_df: pd.DataFrame,
    ) -> dict:
        if self.assets.prototype_router is None:
            return {}

        return dict(self.assets.prototype_router.predict_prefix(prefix_df))

    # ------------------------------------------------------------
    # 3) Build current decision state
    # ------------------------------------------------------------
    def build_current_state(
        self,
        prefix_df: pd.DataFrame,
        prototype_signals: Optional[dict] = None,
    ) -> pd.DataFrame:

        return build_current_decision_state(
            partial_session_df=prefix_df,
            poi_descriptor_df=self.assets.poi_descriptor_df,
            pair_lookup_df=self.assets.pair_lookup_df,
            poi_df=self.assets.poi_df,
            config=self.assets.config,
            _pair_lookup=self.assets.pair_lookup_dict,
            _poi_coord_map=self.assets.poi_coord_map,
            prototype_signals=prototype_signals,
            recent_k=self.assets.recent_k,
        )

    # ------------------------------------------------------------
    # 4) Retrieve historical decision states
    # ------------------------------------------------------------
    def retrieve_cases(
        self,
        current_state_df: pd.DataFrame,
    ) -> pd.DataFrame:

        return retrieve_similar_decision_states(
            query_state=current_state_df,
            case_base_df=self.assets.decision_state_case_base_df,
            encoder=self.assets.decision_state_encoder,
            retrieval_index=self.assets.decision_state_retrieval_index,
            case_vectors=self.assets.decision_state_case_vectors,
            case_coords=self.assets.decision_state_case_coords,
            config=self.assets.config,
            top_k=self.assets.top_k_retrieved_cases,
            same_prototype_only=False,
            prototype_union_k=self.assets.prototype_union_k,
            exclude_same_session=True,
        )

    # ------------------------------------------------------------
    # 5) Aggregate candidate POIs from retrieved cases
    # ------------------------------------------------------------
    def aggregate_candidates(
        self,
        retrieved_cases_df: pd.DataFrame,
    ) -> pd.DataFrame:

        return aggregate_candidate_pois_from_retrieved_cases(
            retrieved_cases_df=retrieved_cases_df,
            top_k_candidates=self.assets.top_k_candidates,
            config=self.assets.config,
        )

    # ------------------------------------------------------------
    # 6) Build evidence block + prompt
    # ------------------------------------------------------------
    def build_prompt_payload(
        self,
        prefix_df: pd.DataFrame,
        current_state_df: pd.DataFrame,
        retrieved_cases_df: pd.DataFrame,
        candidate_pois_df: pd.DataFrame,
    ) -> dict:

        q = current_state_df.iloc[0]
        proto_caption = None

        if self.assets.prototype_caption_map is not None:
            proto_id = q.get("proto_prototype_id", None)
            if proto_id in self.assets.prototype_caption_map:
                proto_caption = self.assets.prototype_caption_map[proto_id]

        evidence_block = build_prompt_ready_evidence_block(
            partial_session_df=prefix_df,
            current_state_df=current_state_df,
            retrieved_cases_df=retrieved_cases_df,
            candidate_pois_df=candidate_pois_df,
            poi_meta_df=self.assets.poi_df,
            config=self.assets.config,
            prototype_caption=proto_caption,
            max_recent_stops=self.assets.max_recent_stops,
            max_evidence_cases=self.assets.max_evidence_cases,
            max_candidates=self.assets.max_candidates,
        )

        prompt_payload = build_llm_reranking_prompt(
            evidence_block=evidence_block,
            candidate_pois_df=candidate_pois_df,
            poi_meta_df=self.assets.poi_df,
            max_candidates=self.assets.max_candidates,
            top_k_return=self.assets.top_k_return,
        )

        prompt_payload["evidence_block"] = evidence_block
        return prompt_payload

    # ------------------------------------------------------------
    # 7) LLM top-1 prediction
    # ------------------------------------------------------------
    def predict_next_poi_from_prefix(
        self,
        prefix_df: pd.DataFrame,
    ) -> dict:
        prototype_signals = self.infer_prototype_signals(prefix_df)
        current_state_df = build_current_decision_state(
            partial_session_df=prefix_df,
            poi_descriptor_df=self.assets.poi_descriptor_df,
            pair_lookup_df=self.assets.pair_lookup_df,
            poi_df=self.assets.poi_df,
            config=self.assets.config,
            _pair_lookup=self.assets.pair_lookup_dict,
            _poi_coord_map=self.assets.poi_coord_map,
            prototype_signals=prototype_signals,
            recent_k=self.assets.recent_k,
        )
        retrieved_cases_df = self.retrieve_cases(current_state_df)
        candidate_pois_df = self.aggregate_candidates(retrieved_cases_df)
        prompt_payload = self.build_prompt_payload(
            prefix_df=prefix_df,
            current_state_df=current_state_df,
            retrieved_cases_df=retrieved_cases_df,
            candidate_pois_df=candidate_pois_df,
        )

        system_prompt = prompt_payload["system_prompt"]
        user_prompt = prompt_payload["user_prompt"]

        llm_text = self.llm_generate_fn(system_prompt, user_prompt)

        fallback_ids = candidate_pois_df["next_POIId"].tolist()
        selected_poi_id = self.llm_parse_fn(llm_text, fallback_ids)

        return {
            "selected_poi_id": selected_poi_id,
            "current_state_df": current_state_df,
            "retrieved_cases_df": retrieved_cases_df,
            "candidate_pois_df": candidate_pois_df,
            "prompt_payload": prompt_payload,
            "llm_raw_text": llm_text,
        }

    # ------------------------------------------------------------
    # 8) Predict from a full held-out session
    # ------------------------------------------------------------
    def predict_from_full_test_session(
        self,
        full_session_df: pd.DataFrame,
    ) -> dict:
        prefix_df, gold_next = self.build_test_query_from_full_session(full_session_df)
        pred = self.predict_next_poi_from_prefix(prefix_df)

        pred["gold_next_poi_id"] = gold_next[self.assets.config.poi_id_col]
        pred["is_correct_at_1"] = pred["selected_poi_id"] == pred["gold_next_poi_id"]
        return pred

    def predict_batch_from_test_checkins(
        self,
        test_checkins_df: pd.DataFrame,
        *,
        min_checkins: int = 2,
        include_details: bool = False,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Run next-POI prediction once per session in a long check-in table.

        ``test_checkins_df`` must contain ``config.session_id_col`` and
        ``config.timestamp_col`` (same schema as training check-ins). Rows are
        grouped by session, sorted by time, then the last check-in is held out
        as the label (same contract as ``predict_from_full_test_session``).

        Parameters
        ----------
        test_checkins_df
            All test sessions concatenated (one row per check-in).
        min_checkins
            Sessions with fewer rows are recorded as ``skipped=True`` (no API call).
        include_details
            If True, adds an ``llm_raw_text`` column per session (can be large).
        show_progress
            If True and ``tqdm`` is installed, show a progress bar over sessions.

        Returns
        -------
        DataFrame with one row per session: session id, gold/selected ids,
        ``is_correct_at_1``, ``skipped``, ``n_checkins``, optional ``error``.
        """
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None  # type: ignore[assignment]

        config = self.assets.config
        sid_col = config.session_id_col
        ts_col = config.timestamp_col

        if sid_col not in test_checkins_df.columns:
            raise ValueError(
                f"test_checkins_df must contain session column {sid_col!r} "
                "(set config.session_id_col to match your CSV)."
            )
        if ts_col not in test_checkins_df.columns:
            raise ValueError(
                f"test_checkins_df must contain timestamp column {ts_col!r} "
                "(set config.timestamp_col to match your CSV)."
            )

        work = test_checkins_df.copy()
        work[ts_col] = pd.to_datetime(work[ts_col], errors="coerce")
        groups = work.sort_values([sid_col, ts_col]).groupby(sid_col, sort=False)

        rows: list[dict[str, Any]] = []
        it = groups
        if show_progress and tqdm is not None:
            it = tqdm(groups, desc="test sessions", unit="session")

        for session_id, session_df in it:
            session_df = session_df.reset_index(drop=True)
            if len(session_df) < min_checkins:
                rows.append(
                    {
                        sid_col: session_id,
                        "skipped": True,
                        "n_checkins": int(len(session_df)),
                        "gold_next_poi_id": None,
                        "selected_poi_id": None,
                        "is_correct_at_1": None,
                        "error": f"fewer than {min_checkins} check-ins",
                    }
                )
                continue

            try:
                pred = self.predict_from_full_test_session(session_df)
                rec: dict[str, Any] = {
                    sid_col: session_id,
                    "skipped": False,
                    "n_checkins": int(len(session_df)),
                    "gold_next_poi_id": pred["gold_next_poi_id"],
                    "selected_poi_id": pred["selected_poi_id"],
                    "is_correct_at_1": bool(pred["is_correct_at_1"]),
                    "error": None,
                }
                if include_details:
                    rec["llm_raw_text"] = pred.get("llm_raw_text", "")
                rows.append(rec)
            except Exception as e:
                rows.append(
                    {
                        sid_col: session_id,
                        "skipped": False,
                        "n_checkins": int(len(session_df)),
                        "gold_next_poi_id": None,
                        "selected_poi_id": None,
                        "is_correct_at_1": None,
                        "error": repr(e),
                    }
                )

        return pd.DataFrame(rows)

    def predict_batch_from_test_checkins_batched_llm(
        self,
        test_checkins_df: pd.DataFrame,
        *,
        llm_batch_generate_fn: Callable[[list[str], list[str]], list[str]],
        min_checkins: int = 2,
        include_details: bool = False,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Batch version of ``predict_batch_from_test_checkins`` that batches *only*
        the LLM generation step.

        It first builds prompts (retrieval + candidates + evidence) per session,
        then calls ``llm_batch_generate_fn(system_prompts, user_prompts)`` once.

        This is particularly efficient with vLLM, which is optimized for batched
        decoding.
        """
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None  # type: ignore[assignment]

        config = self.assets.config
        sid_col = config.session_id_col
        ts_col = config.timestamp_col

        if sid_col not in test_checkins_df.columns:
            raise ValueError(
                f"test_checkins_df must contain session column {sid_col!r} "
                "(set config.session_id_col to match your CSV)."
            )
        if ts_col not in test_checkins_df.columns:
            raise ValueError(
                f"test_checkins_df must contain timestamp column {ts_col!r} "
                "(set config.timestamp_col to match your CSV)."
            )

        work = test_checkins_df.copy()
        work[ts_col] = pd.to_datetime(work[ts_col], errors="coerce")
        groups = work.sort_values([sid_col, ts_col]).groupby(sid_col, sort=False)

        # 1) Build prompts + metadata per session
        session_rows: list[dict[str, Any]] = []
        system_prompts: list[str] = []
        user_prompts: list[str] = []
        fallback_ids_list: list[list[Any]] = []

        it = groups
        if show_progress and tqdm is not None:
            it = tqdm(groups, desc="build prompts", unit="session")

        for session_id, session_df in it:
            session_df = session_df.reset_index(drop=True)
            if len(session_df) < min_checkins:
                session_rows.append(
                    {
                        sid_col: session_id,
                        "skipped": True,
                        "n_checkins": int(len(session_df)),
                        "gold_next_poi_id": None,
                        "selected_poi_id": None,
                        "is_correct_at_1": None,
                        "error": f"fewer than {min_checkins} check-ins",
                        "_needs_llm": False,
                    }
                )
                continue

            try:
                prefix_df, gold_next = self.build_test_query_from_full_session(session_df)
                prototype_signals = self.infer_prototype_signals(prefix_df)
                current_state_df = self.build_current_state(
                    prefix_df, prototype_signals=prototype_signals
                )
                retrieved_cases_df = self.retrieve_cases(current_state_df)
                candidate_pois_df = self.aggregate_candidates(retrieved_cases_df)
                prompt_payload = self.build_prompt_payload(
                    prefix_df=prefix_df,
                    current_state_df=current_state_df,
                    retrieved_cases_df=retrieved_cases_df,
                    candidate_pois_df=candidate_pois_df,
                )

                system_prompts.append(prompt_payload["system_prompt"])
                user_prompts.append(prompt_payload["user_prompt"])

                fallback_ids = candidate_pois_df["next_POIId"].tolist()
                fallback_ids_list.append(fallback_ids)

                session_rows.append(
                    {
                        sid_col: session_id,
                        "skipped": False,
                        "n_checkins": int(len(session_df)),
                        "gold_next_poi_id": gold_next[self.assets.config.poi_id_col],
                        "selected_poi_id": None,
                        "is_correct_at_1": None,
                        "error": None,
                        "_needs_llm": True,
                    }
                )
            except Exception as e:
                session_rows.append(
                    {
                        sid_col: session_id,
                        "skipped": False,
                        "n_checkins": int(len(session_df)),
                        "gold_next_poi_id": None,
                        "selected_poi_id": None,
                        "is_correct_at_1": None,
                        "error": repr(e),
                        "_needs_llm": False,
                    }
                )

        # 2) Batched generation
        if system_prompts:
            llm_texts = llm_batch_generate_fn(system_prompts, user_prompts)
            if len(llm_texts) != len(system_prompts):
                raise RuntimeError(
                    f"llm_batch_generate_fn returned {len(llm_texts)} outputs for "
                    f"{len(system_prompts)} prompts."
                )
        else:
            llm_texts = []

        # 3) Parse outputs back into rows
        out_rows: list[dict[str, Any]] = []
        llm_i = 0
        for rec in session_rows:
            if not rec.pop("_needs_llm"):
                out_rows.append({k: v for k, v in rec.items() if not k.startswith("_")})
                continue

            llm_text = llm_texts[llm_i]
            fallback_ids = fallback_ids_list[llm_i]
            llm_i += 1

            try:
                selected = self.llm_parse_fn(llm_text, fallback_ids)
                rec["selected_poi_id"] = selected
                gold = rec.get("gold_next_poi_id", None)
                rec["is_correct_at_1"] = (selected == gold) if gold is not None else None
                if include_details:
                    rec["llm_raw_text"] = llm_text
            except Exception as e:
                rec["error"] = rec["error"] or repr(e)
                if include_details:
                    rec["llm_raw_text"] = llm_text

            out_rows.append({k: v for k, v in rec.items() if not k.startswith("_")})

        return pd.DataFrame(out_rows), pd.DataFrame({"user_prompt": user_prompts, "llm_responses": llm_texts}, columns=["user_prompt", "llm_responses"])


if __name__ == "__main__":
    pass
