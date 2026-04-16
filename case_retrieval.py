import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from features_extraction import build_feature_blocks

# =====================================================
# 配置
# =====================================================

TOP_M = 3                     # 使用 top-M prototype
PER_PROTO_LIMIT = 50          # 每个 prototype 最多取多少条历史 session
TOP_K = 10                    # 输出多少候选案例
RANDOM_SEED = 42


# =====================================================
# 基础工具
# =====================================================

def build_session_dict(df):
    result = {}
    for sid, g in df.groupby("SessionId"):
        g = g.sort_values("session_checkin_index").reset_index(drop=True)
        result[str(sid)] = g
    return result


def split_partial_and_target(session_df):
    if len(session_df) < 2:
        return None, None
    partial_df = session_df.iloc[:-1].copy().reset_index(drop=True)
    target_row = session_df.iloc[-1].copy()
    return partial_df, target_row


def build_partial_samples(session_dict):
    rows = []
    for sid, sdf in session_dict.items():
        partial_df, target_row = split_partial_and_target(sdf)
        if partial_df is None:
            continue
        rows.append({
            "session_id": str(sid),
            "partial_df": partial_df,
            "target_pid": str(target_row["PId"])
        })
    return rows


# =====================================================
# Prototype Candidate Pool
# =====================================================

def get_query_top_prototypes(assign_row):
    cols = [
        "top1_prototype_id",
        "top2_prototype_id",
        "top3_prototype_id"
    ]
    ids = []

    for c in cols:
        if c in assign_row.index and pd.notna(assign_row[c]):
            ids.append(int(assign_row[c]))

    return ids[:TOP_M]


def build_proto_index(assignments):
    """
    prototype_id -> [session ids]
    """
    proto_map = {}

    for _, row in assignments.iterrows():
        pid = int(row["prototype_id"])
        sid = str(row["SessionId"])

        proto_map.setdefault(pid, []).append(sid)

    return proto_map


def retrieve_candidate_sessions(query_sid,
                                query_assign_row,
                                proto_index):
    top_proto = get_query_top_prototypes(query_assign_row)

    cands = []

    for p in top_proto:
        ids = proto_index.get(p, [])
        cands.extend(ids[:PER_PROTO_LIMIT])

    # 去重 + 去掉自己
    uniq = []
    seen = set()

    for sid in cands:
        if sid == query_sid:
            continue
        if sid not in seen:
            seen.add(sid)
            uniq.append(sid)

    return uniq



def build_session_feature_lookup(feature_split):
    """
    把 build_feature_blocks() 的输出转成:
        SessionId -> feature vector

    默认将 blocks 拼接:
        temporal + category_dense + spatial
    （如果你想加 category_tfidf 也可改）
    """
    meta = feature_split["meta"].copy()

    parts = [
        feature_split["blocks"]["temporal"],
        feature_split["blocks"]["category_dense"],
        feature_split["blocks"]["spatial"],
    ]

    X = np.hstack(parts)

    meta["SessionId"] = meta["SessionId"].astype(str)

    feat_map = {}
    for i, sid in enumerate(meta["SessionId"]):
        feat_map[str(sid)] = X[i]

    return feat_map


def safe_category_of_next(session_df):
    """
    返回 session 最后一个点的 Category
    """
    return str(session_df.iloc[-1]["Category"])


def safe_coord_of_next(session_df):
    """
    返回 session 最后一个点坐标
    自动兼容常见列名
    """
    lat_cols = ["Latitude", "Lat", "latitude", "lat"]
    lon_cols = ["Longitude", "Lon", "longitude", "lng", "lon"]

    lat = None
    lon = None

    for c in lat_cols:
        if c in session_df.columns:
            lat = float(session_df.iloc[-1][c])
            break

    for c in lon_cols:
        if c in session_df.columns:
            lon = float(session_df.iloc[-1][c])
            break

    if lat is None or lon is None:
        return None

    return lat, lon


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0

    p1 = np.radians(lat1)
    p2 = np.radians(lat2)

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    )

    return 2 * R * np.arcsin(np.sqrt(a))


def is_positive_candidate(query_target_df,
                          cand_df,
                          distance_threshold_km=1.0):
    """
    正例规则：
        1. next category 相同
        OR
        2. next poi 距离接近
    """
    # category
    cat_q = safe_category_of_next(query_target_df)
    cat_c = safe_category_of_next(cand_df)

    if cat_q == cat_c:
        return True

    # distance
    coord_q = safe_coord_of_next(query_target_df)
    coord_c = safe_coord_of_next(cand_df)

    if coord_q is None or coord_c is None:
        return False

    dist = haversine_km(
        coord_q[0], coord_q[1],
        coord_c[0], coord_c[1]
    )

    return dist <= distance_threshold_km


# =====================================================
# 修改后的 build_binary_dataset
# =====================================================

def build_binary_dataset(samples,
                         full_session_dict,
                         history_assignments,
                         proto_index,
                         session_feature_map,
                         neg_per_query=5,
                         distance_threshold_km=1.0):
    """
    输入:
        session_feature_map:
            SessionId -> feature vector
            来自 build_feature_blocks()

    标签:
        positive if
            same next category
            OR next location within threshold
    """
    assign_map = history_assignments.set_index("SessionId")
    assign_map.index = assign_map.index.astype(str)

    X = []
    y = []

    used = 0
    skipped = 0

    for item in samples:
        qsid = str(item["session_id"])

        if qsid not in assign_map.index:
            skipped += 1
            continue

        if qsid not in session_feature_map:
            skipped += 1
            continue

        qrow = assign_map.loc[qsid]

        # 完整 query session（含 target）
        full_query_df = full_session_dict.get(qsid, None)
        if full_query_df is None:
            skipped += 1
            continue

        candidate_ids = retrieve_candidate_sessions(
            qsid, qrow, proto_index
        )

        positives = []
        negatives = []

        qvec = session_feature_map[qsid]

        for sid in candidate_ids:
            if sid not in session_feature_map:
                continue

            cand_df = full_session_dict[sid]
            cvec = session_feature_map[sid]

            sim = np.dot(qvec, cvec) / (norm(qvec) * norm(cvec) + 1e-9)

            pair_feat = np.concatenate([
                qvec,
                cvec,
                np.abs(qvec - cvec),
                [sim]
            ])

            label = is_positive_candidate(
                full_query_df,
                cand_df,
                distance_threshold_km=distance_threshold_km
            )

            if label:
                positives.append(pair_feat)
            else:
                negatives.append(pair_feat)

        if len(positives) == 0:
            skipped += 1
            continue

        used += 1

        for feat in positives:
            X.append(feat)
            y.append(1)

        np.random.shuffle(negatives)
        for feat in negatives[:neg_per_query]:
            X.append(feat)
            y.append(0)

    print(f"usable queries={used}, skipped={skipped}")

    return np.array(X), np.array(y)

# =====================================================
# 训练 MLP
# =====================================================

def train_classifier(X, y):
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=300,
            random_state=RANDOM_SEED
        ))
    ])

    clf.fit(X, y)
    return clf


# =====================================================
# 修改后的 predict_one
# =====================================================

def predict_one(query_item,
                full_session_dict,
                history_assignments,
                proto_index,
                model,
                session_feature_map):
    qsid = str(query_item["session_id"])

    assign_map = history_assignments.set_index("SessionId")
    assign_map.index = assign_map.index.astype(str)

    if qsid not in assign_map.index:
        return []

    if qsid not in session_feature_map:
        return []

    qrow = assign_map.loc[qsid]
    qvec = session_feature_map[qsid]

    candidate_ids = retrieve_candidate_sessions(
        qsid, qrow, proto_index
    )

    scored = []

    for sid in candidate_ids:
        if sid not in session_feature_map:
            continue

        cvec = session_feature_map[sid]

        sim = np.dot(qvec, cvec) / (norm(qvec) * norm(cvec) + 1e-9)

        feat = np.concatenate([
            qvec,
            cvec,
            np.abs(qvec - cvec),
            [sim]
        ]).reshape(1, -1)

        prob = model.predict_proba(feat)[0][1]
        scored.append((sid, prob))

    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:TOP_K]


# =====================================================
# 修改后的 run_prediction
# =====================================================

def run_prediction(test_samples,
                   full_session_dict,
                   history_assignments,
                   proto_index,
                   model,
                   session_feature_map):
    rows = []

    for item in test_samples:
        qsid = str(item["session_id"])
        target = item["target_pid"]

        ranked = predict_one(
            item,
            full_session_dict,
            history_assignments,
            proto_index,
            model,
            session_feature_map
        )

        for rank, (sid, score) in enumerate(ranked, 1):
            pred_next = str(full_session_dict[sid].iloc[-1]["PId"])

            rows.append({
                "query_session": qsid,
                "true_next_poi": target,
                "rank": rank,
                "candidate_session": sid,
                "candidate_next_poi": pred_next,
                "score": float(score)
            })

    return pd.DataFrame(rows)


import json
import pandas as pd
import numpy as np


def to_jsonable(v):
    """
    解决 json.dumps 无法处理:
        Timestamp / numpy.int64 / numpy.float64 / NaN
    """
    if pd.isna(v):
        return None

    if isinstance(v, pd.Timestamp):
        return v.isoformat()

    if isinstance(v, (np.integer,)):
        return int(v)

    if isinstance(v, (np.floating,)):
        return float(v)

    if isinstance(v, (np.bool_,)):
        return bool(v)

    return v


def session_to_records(session_df):
    """
    DataFrame -> list[dict]
    自动转 JSON 可序列化格式
    """
    rows = []

    for _, r in session_df.iterrows():
        row = {}

        for k, v in r.to_dict().items():
            row[k] = to_jsonable(v)

        rows.append(row)

    return rows


def build_llm_dataset(test_samples,
                      full_session_dict,
                      history_assignments,
                      proto_index,
                      model,
                      session_feature_map,
                      save_path="module3_llm_dataset.jsonl"):
    """
    输出 jsonl，每行一个 query 样本:

    {
      query_session_id,
      query_partial,
      ground_truth_next_poi,
      retrieved_cases:[...]
    }
    """

    rows = []

    for item in test_samples:
        qsid = str(item["session_id"])
        target_pid = str(item["target_pid"])
        partial_df = item["partial_df"]

        ranked = predict_one(
            item,
            full_session_dict,
            history_assignments,
            proto_index,
            model,
            session_feature_map
        )

        retrieved_cases = []

        for rank, (sid, score) in enumerate(ranked, 1):
            cand_df = full_session_dict[sid]

            retrieved_cases.append({
                "rank": int(rank),
                "candidate_session_id": str(sid),
                "score": float(score),
                "candidate_next_poi": str(cand_df.iloc[-1]["PId"]),
                "trajectory": session_to_records(cand_df)
            })

        sample = {
            "query_session_id": qsid,
            "query_partial": session_to_records(partial_df),
            "ground_truth_next_poi": target_pid,
            "retrieved_cases": retrieved_cases
        }

        rows.append(sample)

    # 保存 jsonl
    with open(save_path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Saved LLM dataset -> {save_path}")

    return rows

def build_partial_checkins_df(checkins_df):
    """
    将完整 session checkins 转为 partial checkins：
    对每个 SessionId，删除最后一个 check-in，仅保留前 n-1 个点。

    输入:
        checkins_df:
            至少包含
            - SessionId
            - session_checkin_index

    输出:
        partial_df:
            与原表结构一致，但每个 session 去掉最后一个点
    """
    keep_parts = []

    for sid, g in checkins_df.groupby("SessionId"):
        g = g.sort_values("session_checkin_index").copy()

        # session 长度不足2，无法构造 partial
        if len(g) < 2:
            continue

        part = g.iloc[:-1].copy()
        keep_parts.append(part)

    if len(keep_parts) == 0:
        return checkins_df.iloc[0:0].copy()

    partial_df = pd.concat(keep_parts, axis=0).reset_index(drop=True)

    return partial_df

# =====================================================
# Main
# =====================================================

def main():
    data_dir = Path(
        "preprocessed_data/NYC/canonical_gap6h_minlen3_split701020"
    )

    # checkins
    train_df = pd.read_csv(
        data_dir / "step3_train_checkins.csv",
        parse_dates=["Time"]
    )

    val_df = pd.read_csv(
        data_dir / "step3_validation_checkins.csv",
        parse_dates=["Time"]
    )

    test_df = pd.read_csv(
        data_dir / "step3_test_checkins.csv",
        parse_dates=["Time"]
    )

    # session dict
    train_sessions = build_session_dict(train_df)
    val_sessions = build_session_dict(val_df)
    test_sessions = build_session_dict(test_df)

    all_sessions = {}
    all_sessions.update(train_sessions)
    all_sessions.update(val_sessions)
    all_sessions.update(test_sessions)

    with open(data_dir / "module_1_gmm_data.pkl", "rb") as f:
        gmm = pickle.load(f)

    train_assignments = gmm["train"]["assignments"]
    test_assignments = gmm["test"]["assignments"]
    val_assignments = gmm["val"]["assignments"]

    assignments = [train_assignments, test_assignments, val_assignments]
    assignments = pd.concat(assignments, axis=0).reset_index(drop=True)

    proto_index = build_proto_index(assignments)

    train_samples = build_partial_samples(train_sessions)

    feature_data = build_feature_blocks(
        train_checkins=build_partial_checkins_df(train_df),
        val_checkins=build_partial_checkins_df(val_df),
        test_checkins=build_partial_checkins_df(test_df),
        region_col=None,
        h3_resolution=7,
        category_ngram_range=(1, 2),
        category_svd_components=64,
        random_state=42,
    )

    session_feature_map = {}
    session_feature_map.update(build_session_feature_lookup(feature_data["train"]))
    session_feature_map.update(build_session_feature_lookup(feature_data["test"]))
    session_feature_map.update(build_session_feature_lookup(feature_data["val"]))

    # 原 build_binary_dataset 调用替换为：
    X_train, y_train = build_binary_dataset(
        train_samples,
        all_sessions,
        assignments,
        proto_index,
        session_feature_map=session_feature_map,
        neg_per_query=5,
        distance_threshold_km=1.0
    )

    print("X_train:", X_train.shape)
    print("Positive ratio:", y_train.mean())

    # train mlp
    model = train_classifier(X_train, y_train)

    llm_rows = build_llm_dataset(
        test_samples=train_samples,
        full_session_dict=all_sessions,
        history_assignments=assignments,
        proto_index=proto_index,
        model=model,
        session_feature_map=session_feature_map,
        save_path="module3_llm_val_dataset.jsonl"
    )

    # test
    test_samples = build_partial_samples(test_sessions)

    llm_rows = build_llm_dataset(
        test_samples=test_samples,
        full_session_dict=all_sessions,
        history_assignments=assignments,
        proto_index=proto_index,
        model=model,
        session_feature_map=session_feature_map,
        save_path="module3_llm_test_dataset.jsonl"
    )

    val_samples = build_partial_samples(val_sessions)
    llm_rows = build_llm_dataset(
        test_samples=val_samples,
        full_session_dict=all_sessions,
        history_assignments=assignments,
        proto_index=proto_index,
        model=model,
        session_feature_map=session_feature_map,
        save_path="module3_llm_val_dataset.jsonl"
    )



if __name__ == "__main__":
    main()