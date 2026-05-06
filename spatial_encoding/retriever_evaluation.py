import pickle
import pandas as pd
import numpy as np
from typing import Any, Iterable
from tqdm import tqdm
from termcolor import cprint
from pathlib import Path

from extract_poi_spatial_descriptors import SpatialEncodingConfig
from retrieve_decisions_states import (
    build_retrieval_index,
    DecisionStateEncoder,
    DecisionStateRetrievalIndex,
    build_current_decision_state,
)
from retrieve_candidates_pois import retrieve_candidate_next_pois


def _normalize_poi_id(value: Any) -> Any:
    """
    Normalize POI ids for equality.
    """
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return int(value) if float(value).is_integer() else float(value)
    return str(value)


def _rank_of_gold(candidate_ids: Iterable[Any], gold_poi_id: Any) -> int | None:
    gold = _normalize_poi_id(gold_poi_id)
    for i, candidate_id in enumerate(candidate_ids, start=1):
        if _normalize_poi_id(candidate_id) == gold:
            return i
    return None


def _prepare_lookup_df(pair_lookup_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure pair lookup is indexed by (src_POIId, dst_POIId), as expected by
    compute_single_session_transitions().
    """
    if isinstance(pair_lookup_df.index, pd.MultiIndex):
        return pair_lookup_df

    required = ["src_POIId", "dst_POIId"]
    missing = [col for col in required if col not in pair_lookup_df.columns]
    if missing:
        raise ValueError(f"pair_lookup_df missing required columns: {missing}")

    return pair_lookup_df.drop_duplicates(subset=required, keep="first").set_index(required).copy()


def _prepare_coord_df(poi_df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Ensure POI coordinates are indexed by POI id, as expected by
    compute_single_session_transitions().
    """
    required = [config.poi_id_col, config.lat_col, config.lon_col]
    missing = [col for col in required if col not in poi_df.columns]
    if missing:
        raise ValueError(f"poi_df missing required columns: {missing}")

    return (
        poi_df[required]
        .drop_duplicates(subset=[config.poi_id_col], keep="first")
        .set_index(config.poi_id_col)
        .copy()
    )


def evaluate_candidate_retriever(
    test_checkins_df: pd.DataFrame,
    *,
    poi_descriptor_df: pd.DataFrame,
    lookup_df: pd.DataFrame,
    coord_df: pd.DataFrame,
    encoder: DecisionStateEncoder,
    retrieval_index: DecisionStateRetrievalIndex,
    config,
    k_values: tuple[int, ...] = (1, 3, 5, 10, 20),
    top_k_cases: int = 50,
    top_m_pois: int | None = None,
    temperature: float = 0.2,
    same_prototype_only: bool = False,
    exclude_same_session: bool = True,
    prototype_union_k: int = 3,
    min_checkins: int = 2,
    max_sessions: int | None = None,
    random_state: int = 42,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate candidate retrieval by holding out each session's last POI.

    For each test session:
      1. use all but the last check-in as the observed prefix,
      2. retrieve historical decision states,
      3. aggregate candidate next POIs,
      4. check whether the held-out POI appears in the top-k candidates.

    Returns
    -------
    metrics_df:
        One-row summary with hit@k / recall@k, MRR, mean rank, and diagnostics.
    details_df:
        One row per evaluated session with gold id, rank, top candidates, and errors.
    """
    if not k_values:
        raise ValueError("k_values cannot be empty")

    k_values = tuple(sorted({int(k) for k in k_values if int(k) > 0}))
    if not k_values:
        raise ValueError("k_values must contain at least one positive integer")

    top_m_pois = max(k_values) if top_m_pois is None else int(top_m_pois)
    top_m_pois = max(top_m_pois, max(k_values))

    required_cols = [config.session_id_col, config.timestamp_col, config.poi_id_col]
    missing = [col for col in required_cols if col not in test_checkins_df.columns]
    if missing:
        raise ValueError(f"test_checkins_df missing required columns: {missing}")

    sort_cols = [config.session_id_col, config.timestamp_col, config.poi_id_col]
    work = test_checkins_df.copy()
    work[config.timestamp_col] = pd.to_datetime(work[config.timestamp_col], errors="coerce")
    work = work.loc[work[config.timestamp_col].notna()].copy()
    work = work.sort_values(sort_cols).reset_index(drop=True)

    session_ids = work[config.session_id_col].drop_duplicates().tolist()
    if max_sessions is not None and len(session_ids) > int(max_sessions):
        rng = np.random.default_rng(int(random_state))
        session_ids = rng.choice(session_ids, size=int(max_sessions), replace=False).tolist()
        keep = set(session_ids)
        work = work.loc[work[config.session_id_col].isin(keep)].copy()

    groups = work.groupby(config.session_id_col, sort=False)
    iterator = groups
    if show_progress:
        iterator = tqdm(groups, desc="evaluate retriever", unit="session")

    rows: list[dict[str, Any]] = []
    for session_id, session_df in iterator:
        session_df = session_df.sort_values([config.timestamp_col, config.poi_id_col]).reset_index(drop=True)

        base_row = {
            config.session_id_col: session_id,
            "n_checkins": int(len(session_df)),
            "skipped": False,
            "error": None,
        }

        if len(session_df) < int(min_checkins):
            rows.append(
                {
                    **base_row,
                    "skipped": True,
                    "error": f"fewer than {min_checkins} check-ins",
                    "gold_next_POIId": None,
                    "gold_rank": None,
                    "candidate_count": 0,
                    "retrieved_case_count": 0,
                    "top_candidates": [],
                }
            )
            continue

        prefix_df = session_df.iloc[:-1].copy().reset_index(drop=True)
        gold_next = session_df.iloc[-1]
        gold_poi_id = gold_next[config.poi_id_col]

        try:
            query_state = build_current_decision_state(
                partial_session_df=prefix_df,
                poi_descriptor_df=poi_descriptor_df,
                lookup_df=lookup_df,
                coord_df=coord_df,
                config=config,
            )

            result = retrieve_candidate_next_pois(
                query_state=query_state,
                encoder=encoder,
                config=config,
                retrieval_index=retrieval_index,
                top_k_cases=top_k_cases,
                top_m_pois=top_m_pois,
                same_prototype_only=same_prototype_only,
                exclude_same_session=exclude_same_session,
                prototype_union_k=prototype_union_k,
                temperature=temperature,
            )

            retrieved_cases = result["retrieved_cases"]
            candidate_pois = result["candidate_pois"]
            candidate_ids = candidate_pois["next_POIId"].tolist()
            rank = _rank_of_gold(candidate_ids, gold_poi_id)

            rec = {
                **base_row,
                "gold_next_POIId": gold_poi_id,
                "gold_rank": rank,
                "candidate_count": int(len(candidate_pois)),
                "retrieved_case_count": int(len(retrieved_cases)),
                "top_candidates": candidate_ids[: max(k_values)],
            }
            for k in k_values:
                rec[f"hit@{k}"] = bool(rank is not None and rank <= k)
                rec[f"recall@{k}"] = float(rec[f"hit@{k}"])
            rows.append(rec)

        except Exception as e:
            rec = {
                **base_row,
                "error": repr(e),
                "gold_next_POIId": gold_poi_id,
                "gold_rank": None,
                "candidate_count": 0,
                "retrieved_case_count": 0,
                "top_candidates": [],
            }
            for k in k_values:
                rec[f"hit@{k}"] = False
                rec[f"recall@{k}"] = 0.0
            rows.append(rec)

    details_df = pd.DataFrame(rows)
    if details_df.empty:
        summary: dict[str, Any] = {
            "n_sessions_total": 0,
            "n_sessions_evaluated": 0,
            "n_sessions_skipped": 0,
            "n_sessions_error": 0,
            "top_k_cases": int(top_k_cases),
            "top_m_pois": int(top_m_pois),
            "temperature": float(temperature),
            "mrr": np.nan,
            "mean_gold_rank": np.nan,
            "coverage": np.nan,
        }
        for k in k_values:
            summary[f"hit@{k}"] = np.nan
            summary[f"recall@{k}"] = np.nan
        return pd.DataFrame([summary]), details_df

    valid_mask = (~details_df["skipped"].fillna(False)) & details_df["error"].isna()
    valid = details_df.loc[valid_mask].copy()

    summary: dict[str, Any] = {
        "n_sessions_total": int(len(details_df)),
        "n_sessions_evaluated": int(len(valid)),
        "n_sessions_skipped": int(details_df["skipped"].fillna(False).sum()),
        "n_sessions_error": int(((~details_df["skipped"].fillna(False)) & details_df["error"].notna()).sum()),
        "top_k_cases": int(top_k_cases),
        "top_m_pois": int(top_m_pois),
        "temperature": float(temperature),
    }

    if len(valid) == 0:
        for k in k_values:
            summary[f"hit@{k}"] = np.nan
            summary[f"recall@{k}"] = np.nan
        summary["mrr"] = np.nan
        summary["mean_gold_rank"] = np.nan
        summary["coverage"] = np.nan
    else:
        ranks = pd.to_numeric(valid["gold_rank"], errors="coerce")
        for k in k_values:
            summary[f"hit@{k}"] = float(valid[f"hit@{k}"].mean())
            summary[f"recall@{k}"] = summary[f"hit@{k}"]
        summary["mrr"] = float((1.0 / ranks.dropna()).sum() / len(valid))
        summary["mean_gold_rank"] = float(ranks.dropna().mean()) if ranks.notna().any() else np.nan
        summary["coverage"] = float((valid["candidate_count"] > 0).mean())
        summary["mean_candidate_count"] = float(valid["candidate_count"].mean())
        summary["mean_retrieved_case_count"] = float(valid["retrieved_case_count"].mean())

    metrics_df = pd.DataFrame([summary])
    return metrics_df, details_df


if __name__ == "__main__":
    city = "nyc"
    # max_eval_sessions = 200
    k_values = (1, 3, 5, 10, 20)

    scrip_dir = Path(__file__).resolve().parent.parent
    decision_state_table_df = pd.read_csv(scrip_dir / f"artifacts/{city}/{city}_decision_state_table.csv")

    poi_descriptor_df = pd.read_csv(scrip_dir / f"artifacts/{city}/{city}_poi_descriptor.csv")

    with open(scrip_dir / f"artifacts/{city}/{city}_pair_lookup.pkl", "rb") as f:
        pair_lookup = pickle.load(f)
    with open(scrip_dir / f"artifacts/{city}/{city}_poi_coord_map.pkl", "rb") as f:
        poi_coord_map = pickle.load(f)
    lookup_df = pd.DataFrame.from_dict(pair_lookup, orient="index")
    lookup_df.index = pd.MultiIndex.from_tuples(lookup_df.index, names=["src_POIId", "dst_POIId"])  # fmt: skip
    coord_df = pd.DataFrame.from_dict(poi_coord_map, orient="index")

    config = SpatialEncodingConfig()

    encoder = DecisionStateEncoder(config=config)
    encoder.fit(decision_state_table_df)
    case_vectors = encoder.transform(decision_state_table_df)
    case_coords = encoder.extract_coords(decision_state_table_df)

    retrieval_index = build_retrieval_index(
        case_base_df=decision_state_table_df,
        case_vectors=case_vectors,
        config=config,
        case_coords=case_coords,
    )

    sid_col = config.session_id_col
    ts_col = config.timestamp_col
    poi_id_col = config.poi_id_col

    test_checkins = pd.read_csv(scrip_dir / f"data/{city}/test_sample.csv")
    test_checkins = test_checkins.rename(columns={"pseudo_session_trajectory_id": sid_col})
    test_checkins[ts_col] = pd.to_datetime(test_checkins[ts_col], errors="coerce")
    test_checkins = test_checkins.sort_values([sid_col, ts_col, poi_id_col]).reset_index(drop=True)

    metrics_df, details_df = evaluate_candidate_retriever(
        test_checkins_df=test_checkins,
        poi_descriptor_df=poi_descriptor_df,
        lookup_df=lookup_df,
        coord_df=coord_df,
        encoder=encoder,
        retrieval_index=retrieval_index,
        config=config,
        k_values=k_values,
        top_k_cases=50,
        top_m_pois=max(k_values),
        temperature=0.2,
        max_sessions=None,
        show_progress=True,
    )

    print("\nRetriever candidate metrics:")
    print(metrics_df.to_string(index=False))

    out_dir = scrip_dir / f"artifacts/{city}"
    metrics_path = out_dir / f"{city}_candidate_retriever_metrics.csv"
    details_path = out_dir / f"{city}_candidate_retriever_details.csv"
    metrics_df.to_csv(metrics_path, index=False)
    details_df.to_csv(details_path, index=False)
    cprint(f"Wrote metrics to {metrics_path}", "green")
    cprint(f"Wrote details to {details_path}", "green")
