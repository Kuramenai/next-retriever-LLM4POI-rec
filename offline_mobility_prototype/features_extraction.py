import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, Normalizer, normalize
from temporal_features_extraction import (
    fit_duration_normalizer_from_checkins,
    build_temporal_feature_matrix,
)
from categorical_features_extraction import build_category_documents
from spatial_features_extraction import build_session_spatial_aggregates
from termcolor import cprint


def align_session_dataframe(
    base_meta: pd.DataFrame, other_df: pd.DataFrame, name: str
) -> pd.DataFrame:
    """
    Align another session-level dataframe to the SessionId order of base_meta.
    """
    if "SessionId" not in base_meta.columns or "SessionId" not in other_df.columns:
        raise ValueError("Both dataframes must contain 'SessionId'")

    aligned = base_meta[["SessionId"]].merge(
        other_df,
        on="SessionId",
        how="left",
        validate="one_to_one",
    )

    if aligned.isnull().any().any():
        missing_sessions = aligned.loc[
            aligned.isnull().any(axis=1), "SessionId"
        ].tolist()
        raise ValueError(
            f"{name} is missing rows for some sessions. "
            f"Missing SessionIds (first 10 shown): {missing_sessions[:10]}"
        )

    return aligned


def build_feature_blocks(
    train_checkins: pd.DataFrame,
    val_checkins: pd.DataFrame,
    test_checkins: pd.DataFrame,
    *,
    region_col: str | None = None,
    h3_resolution: int = 8,
    category_ngram_range: tuple[int, int] = (1, 2),
    category_svd_components: int | None = 64,
    random_state: int = 42,
) -> dict:
    """
    Build Module-1 feature blocks and final dense matrices for GMM.

    Expected check-in columns:
      - SessionId
      - Time
      - PId
      - Category
      - Latitude
      - Longitude

    Returns a dictionary with:
      - train / val / test dense matrices
      - aligned session metadata
      - individual blocks
      - fitted preprocessing artifacts
    """

    # ------------------------------------------------------------------
    # 1) TEMPORAL BLOCK
    # ------------------------------------------------------------------
    cprint("Extracting temporal features...", "yellow")
    duration_mean, duration_std = fit_duration_normalizer_from_checkins(train_checkins)

    X_train_temp, train_temp_meta = build_temporal_feature_matrix(
        train_checkins,
        duration_mean=duration_mean,
        duration_std=duration_std,
    )
    X_val_temp, val_temp_meta = build_temporal_feature_matrix(
        val_checkins,
        duration_mean=duration_mean,
        duration_std=duration_std,
    )
    X_test_temp, test_temp_meta = build_temporal_feature_matrix(
        test_checkins,
        duration_mean=duration_mean,
        duration_std=duration_std,
    )

    cprint("Temporal features extracted successfully.", "green")

    # ------------------------------------------------------------------
    # 2) CATEGORY BLOCK
    # ------------------------------------------------------------------
    # fmt: off
    cprint("Extracting category features...", "yellow")
    train_cat_df = build_category_documents(train_checkins)
    val_cat_df = build_category_documents(val_checkins)
    test_cat_df = build_category_documents(test_checkins)

    train_cat_df = align_session_dataframe(train_temp_meta, train_cat_df, name="train_cat_df")
    val_cat_df = align_session_dataframe(val_temp_meta, val_cat_df, name="val_cat_df")
    test_cat_df = align_session_dataframe(test_temp_meta, test_cat_df, name="test_cat_df")

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=category_ngram_range,
        token_pattern=r"(?u)\b\S+\b",
    )

    X_train_cat_tfidf = vectorizer.fit_transform(train_cat_df["category_doc"])
    X_val_cat_tfidf = vectorizer.transform(val_cat_df["category_doc"])
    X_test_cat_tfidf = vectorizer.transform(test_cat_df["category_doc"])

    # Per roadmap: L2-normalize TF-IDF block separately
    X_train_cat_tfidf = normalize(X_train_cat_tfidf, norm="l2")
    X_val_cat_tfidf = normalize(X_val_cat_tfidf, norm="l2")
    X_test_cat_tfidf = normalize(X_test_cat_tfidf, norm="l2")

    # GMM needs dense input; compress sparse TF-IDF if requested
    if category_svd_components is not None:
        max_valid_components = min(X_train_cat_tfidf.shape[0] - 1, X_train_cat_tfidf.shape[1] - 1)
        if max_valid_components < 1:
            raise ValueError("Not enough training sessions or category vocabulary to run TruncatedSVD.")

        n_components = min(category_svd_components, max_valid_components)

        category_svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        category_post_normalizer = Normalizer(norm="l2")

        X_train_cat = category_post_normalizer.fit_transform(category_svd.fit_transform(X_train_cat_tfidf))
        X_val_cat = category_post_normalizer.transform(category_svd.transform(X_val_cat_tfidf))
        X_test_cat = category_post_normalizer.transform(category_svd.transform(X_test_cat_tfidf))
    else:
        category_svd = None
        category_post_normalizer = None
        X_train_cat = X_train_cat_tfidf.toarray()
        X_val_cat = X_val_cat_tfidf.toarray()
        X_test_cat = X_test_cat_tfidf.toarray()

    X_train_cat = np.asarray(X_train_cat, dtype=np.float32)
    X_val_cat = np.asarray(X_val_cat, dtype=np.float32)
    X_test_cat = np.asarray(X_test_cat, dtype=np.float32)

    cprint("Category features extracted successfully.", "green")

    # ------------------------------------------------------------------
    # 3) SPATIAL BLOCK
    # ------------------------------------------------------------------
    cprint("Extracting spatial features...", "yellow")
    train_spatial_df = build_session_spatial_aggregates(
        train_checkins,
        region_col=region_col,
        h3_resolution=h3_resolution,
    )
    val_spatial_df = build_session_spatial_aggregates(
        val_checkins,
        region_col=region_col,
        h3_resolution=h3_resolution,
    )
    test_spatial_df = build_session_spatial_aggregates(
        test_checkins,
        region_col=region_col,
        h3_resolution=h3_resolution,
    )

    train_spatial_df = align_session_dataframe(
        train_temp_meta, train_spatial_df, name="train_spatial_df"
    )
    val_spatial_df = align_session_dataframe(
        val_temp_meta, val_spatial_df, name="val_spatial_df"
    )
    test_spatial_df = align_session_dataframe(
        test_temp_meta, test_spatial_df, name="test_spatial_df"
    )

    spatial_cols = [
        "movement_radius_km",
        "h3_entropy",
        "start_end_centroid_displacement_km",
    ]

    spatial_scaler = StandardScaler()
    X_train_spatial = spatial_scaler.fit_transform(train_spatial_df[spatial_cols].to_numpy(dtype=np.float32))
    X_val_spatial = spatial_scaler.transform(val_spatial_df[spatial_cols].to_numpy(dtype=np.float32))
    X_test_spatial = spatial_scaler.transform(test_spatial_df[spatial_cols].to_numpy(dtype=np.float32))

    X_train_spatial = np.asarray(X_train_spatial, dtype=np.float32)
    X_val_spatial = np.asarray(X_val_spatial, dtype=np.float32)
    X_test_spatial = np.asarray(X_test_spatial, dtype=np.float32)

    cprint("Spatial features extracted successfully.", "green")

    # ------------------------------------------------------------------
    # 4) FINAL DENSE MATRICES FOR GMM
    # ------------------------------------------------------------------
    X_train = np.hstack([X_train_temp, X_train_cat, X_train_spatial]).astype(np.float32)
    X_val = np.hstack([X_val_temp, X_val_cat, X_val_spatial]).astype(np.float32)
    X_test = np.hstack([X_test_temp, X_test_cat, X_test_spatial]).astype(np.float32)

    # ------------------------------------------------------------------
    # 5) METADATA
    # ------------------------------------------------------------------
    train_meta = (
        train_temp_meta[["SessionId", "session_start_time", "session_end_time"]]
        .merge(
            train_cat_df[["SessionId", "category_sequence", "category_doc"]],
            on="SessionId",
            how="left",
        )
        .merge(
            train_spatial_df[["SessionId"] + spatial_cols], on="SessionId", how="left"
        )
    )

    val_meta = (
        val_temp_meta[["SessionId", "session_start_time", "session_end_time"]]
        .merge(
            val_cat_df[["SessionId", "category_sequence", "category_doc"]],
            on="SessionId",
            how="left",
        )
        .merge(val_spatial_df[["SessionId"] + spatial_cols], on="SessionId", how="left")
    )

    test_meta = (
        test_temp_meta[["SessionId", "session_start_time", "session_end_time"]]
        .merge(
            test_cat_df[["SessionId", "category_sequence", "category_doc"]],
            on="SessionId",
            how="left",
        )
        .merge(
            test_spatial_df[["SessionId"] + spatial_cols], on="SessionId", how="left"
        )
    )

    return {
        "train": {
            "X": X_train,
            "meta": train_meta,
            "blocks": {
                "temporal": X_train_temp,
                "category_dense": X_train_cat,
                "category_tfidf": X_train_cat_tfidf,
                "spatial": X_train_spatial,
            },
        },
        "val": {
            "X": X_val,
            "meta": val_meta,
            "blocks": {
                "temporal": X_val_temp,
                "category_dense": X_val_cat,
                "category_tfidf": X_val_cat_tfidf,
                "spatial": X_val_spatial,
            },
        },
        "test": {
            "X": X_test,
            "meta": test_meta,
            "blocks": {
                "temporal": X_test_temp,
                "category_dense": X_test_cat,
                "category_tfidf": X_test_cat_tfidf,
                "spatial": X_test_spatial,
            },
        },
        "artifacts": {
            "duration_mean": duration_mean,
            "duration_std": duration_std,
            "category_vectorizer": vectorizer,
            "category_svd": category_svd,
            "category_post_normalizer": category_post_normalizer,
            "spatial_scaler": spatial_scaler,
            "spatial_cols": spatial_cols,
        },
    }