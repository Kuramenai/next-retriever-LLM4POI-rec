import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from termcolor import cprint


def extract_session_next_poi(
    checkins_df: pd.DataFrame,
    session_id_col: str = "SessionId",
    checkin_time_col: str = "CheckinTime",
    poi_id_col: str = "PoiId",
) -> pd.Series:
    """
    Extract the next POI for each session from the check-ins dataframe.
    """
    required_cols = [session_id_col, checkin_time_col, poi_id_col]
    missing_cols = [c for c in required_cols if c not in checkins_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in checkins_df: {missing_cols}")

    df = checkins_df.copy()

    df = df.sort_values(required_cols).reset_index(drop=True)
    last_rows = df.groupby(session_id_col, sort=False).tail(1)
    next_poi_ids = last_rows.set_index(session_id_col)[poi_id_col]
    return next_poi_ids


def next_poi_train_vocab_upper_bound(train_next_poi, query_next_poi):
    cprint("Evaluating next POI train vocabulary upper bound...", "yellow")
    train_set = set(train_next_poi.dropna().tolist())
    q = query_next_poi.dropna()

    seen = q.isin(train_set)

    results_dict = {
        "n_queries": int(len(q)),
        "n_train_unique_next_poi": int(len(train_set)),
        "query_next_poi_seen_in_train_recall_upper_bound": float(seen.mean()),
        "n_seen": int(seen.sum()),
        "n_unseen": int((~seen).sum()),
    }
    return pd.DataFrame(results_dict, index=[0])


def prototype_popularity_recall_at_k(
    query_assignments: pd.DataFrame,
    train_assignments: pd.DataFrame,
    train_next_poi: pd.Series,
    query_next_poi: pd.Series,
    top_m: int = 3,
    ks=(20, 50, 100),
):
    """
    Rank next-POI candidates by frequency inside the query's top-M prototypes.

    """
    cprint(
        f"Evaluating prototype popularity recall at k with top_m={top_m} and ks={ks}...",
        "yellow",
    )
    train_df = train_assignments.copy().merge(
        train_next_poi.rename("next_poi"),
        left_on="SessionId",
        right_index=True,
        how="inner",
        validate="one_to_one",
    )

    proto_to_counter = {}
    for pid, g in train_df.groupby("prototype_id"):
        proto_to_counter[int(pid)] = Counter(g["next_poi"].dropna().tolist())

    query_df = query_assignments.copy().merge(
        query_next_poi.rename("true_next_poi"),
        left_on="SessionId",
        right_index=True,
        how="inner",
        validate="one_to_one",
    )

    hits = {k: [] for k in ks}
    candidate_sizes = []

    for _, row in query_df.iterrows():
        true_poi = row["true_next_poi"]

        score = defaultdict(float)

        for j in range(1, top_m + 1):
            id_col = f"top{j}_prototype_id"
            prob_col = f"top{j}_prototype_prob"

            if id_col not in row:
                continue

            pid = int(row[id_col])
            weight = float(row[prob_col]) if prob_col in row else 1.0

            for poi, cnt in proto_to_counter.get(pid, Counter()).items():
                score[poi] += weight * cnt

        ranked = [
            poi
            for poi, _ in sorted(
                score.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        ]

        candidate_sizes.append(len(ranked))

        for k in ks:
            hits[k].append(true_poi in set(ranked[:k]))

    out = {
        "top_m": top_m,
        "n_queries": int(len(query_df)),
        "avg_candidate_pool_size": float(np.mean(candidate_sizes)),
        "median_candidate_pool_size": float(np.median(candidate_sizes)),
    }

    for k in ks:
        out[f"prototype_popularity_recall@{k}"] = float(np.mean(hits[k]))

    return pd.DataFrame(out, index=[0])


def prototype_pool_next_poi_recall(
    query_assignments: pd.DataFrame,
    train_assignments: pd.DataFrame,
    train_next_poi: pd.Series,
    query_next_poi: pd.Series,
    top_m: int = 3,
) -> dict:
    """
    Measures whether the true query next POI appears among next POIs
    observed in training sessions assigned to the query's top-M prototypes.
    """
    cprint(f"Evaluating prototype pool next POI recall with top_m={top_m}...", "yellow")
    # cprint(f"Number of train assignments: {len(train_assignments)}", "yellow")

    # Attach train labels by SessionId
    train_df = train_assignments.copy()
    train_df = train_df.merge(
        train_next_poi.rename("next_poi"),
        left_on="SessionId",
        right_index=True,
        how="inner",
        validate="one_to_one",
    )

    # cprint(f"Number of train assignments after merging with next POIs: {len(train_df)}", "yellow")

    # Map each hard prototype to the set of next POIs observed in training
    proto_to_next_pois = (
        train_df.groupby("prototype_id")["next_poi"].apply(lambda x: set(x.dropna())).to_dict()
    )

    # Attach query labels by SessionId
    # cprint(f"Number of query assignments: {len(query_assignments)}", "yellow")
    query_df = query_assignments.copy()
    query_df = query_df.merge(
        query_next_poi.rename("true_next_poi"),
        left_on="SessionId",
        right_index=True,
        how="inner",
        validate="one_to_one",
    )
    # cprint(f"Number of query assignments after merging with next POIs: {len(query_df)}", "yellow")

    hits = []
    pool_sizes = []

    # query_proto_ids = pd.DataFrame(query_df["SessionId"], columns=["SessionId"])
    # query_proto_ids["prototype_ids"] = query_df.apply(lambda row: [int(row[f"top{j}_prototype_id"]) for j in range(1, top_m + 1)], axis=1)

    for _, row in query_df.iterrows():
        true_poi = row["true_next_poi"]

        # get prototype ids assigned to each query session
        proto_ids = []
        for j in range(1, top_m + 1):
            col = f"top{j}_prototype_id"
            if col in row:
                proto_ids.append(int(row[col]))

        # get next POIs assigned to each prototype
        pool = set()
        for pid in proto_ids:
            pool |= proto_to_next_pois.get(pid, set())

        hits.append(true_poi in pool)
        pool_sizes.append(len(pool))

    results_dict = {
        "top_m": top_m,
        "n_query_assignments": len(query_assignments),
        "n_queries": len(query_df),
        "prototype_pool_recall": float(np.mean(hits)),
        "n_pool_sizes": len(pool_sizes),
        "avg_unique_next_poi_pool_size": float(np.mean(pool_sizes)),
        "median_unique_next_poi_pool_size": float(np.median(pool_sizes)),
    }

    return pd.DataFrame(results_dict, index=[0])


if __name__ == "__main__":
    city = "nyc"
    scrip_dir = Path(__file__).resolve().parent.parent

    train_checkins = pd.read_csv(scrip_dir / f"data/{city}/train_sample.csv")
    # val_checkins = pd.read_csv(scrip_dir / f"data/{city}/validate_sample_with_traj.csv")
    test_checkins = pd.read_csv(scrip_dir / f"data/{city}/test_sample.csv")

    col_mapping = {
        "pseudo_session_trajectory_id": "SessionId",
    }
    train_checkins = train_checkins.rename(columns=col_mapping)
    # val_checkins = val_checkins.rename(columns=col_mapping)
    test_checkins = test_checkins.rename(columns=col_mapping)

    train_checkins["CheckinTime"] = pd.to_datetime(train_checkins["UTCTimeOffset"])
    # val_checkins["CheckinTime"] = pd.to_datetime(val_checkins["UTCTimeOffset"])
    test_checkins["CheckinTime"] = pd.to_datetime(test_checkins["UTCTimeOffset"])

    with open(scrip_dir / f"artifacts/{city}/{city}_gmm_cluster.pkl", "rb") as f:
        gmm_data = pickle.load(f)

    train_assignments = gmm_data["train"]["assignments"]
    # val_assignments = gmm_data["val"]["assignments"]
    test_assignments = gmm_data["test"]["assignments"]

    train_next_poi = extract_session_next_poi(train_checkins)
    # val_next_poi = extract_session_next_poi(val_checkins)
    test_next_poi = extract_session_next_poi(test_checkins)

    # print(
    #     prototype_pool_next_poi_recall(
    #         query_assignments=test_assignments,
    #         train_assignments=train_assignments,
    #         train_next_poi=train_next_poi,
    #         query_next_poi=test_next_poi,
    #         top_m=3,
    #     )
    # )

    # print(
    #     next_poi_train_vocab_upper_bound(
    #         train_next_poi=train_next_poi,
    #         query_next_poi=test_next_poi,
    #     )
    # )

    print(
        prototype_popularity_recall_at_k(
            query_assignments=test_assignments,
            train_assignments=train_assignments,
            train_next_poi=train_next_poi,
            query_next_poi=test_next_poi,
            top_m=3,
            ks=(20, 50, 100, 200),
        )
    )
