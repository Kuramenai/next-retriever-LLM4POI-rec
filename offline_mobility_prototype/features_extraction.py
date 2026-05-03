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
    base_meta: pd.DataFrame,
    other_df: pd.DataFrame,
    name: str,
    session_id_col: str = "SessionId",
) -> pd.DataFrame:
    """
    Align another session-level dataframe to the SessionId order of base_meta.
    """
    if (
        session_id_col not in base_meta.columns
        or session_id_col not in other_df.columns
    ):
        raise ValueError(
            f"Both dataframes must contain session id column {session_id_col!r}"
        )

    aligned = base_meta[[session_id_col]].merge(
        other_df,
        on=session_id_col,
        how="left",
        validate="one_to_one",
    )

    if aligned.isnull().any().any():
        missing_sessions = aligned.loc[
            aligned.isnull().any(axis=1), session_id_col
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
    # ── New parameters for taxonomy + absorption ──
    taxonomy_level: str = "raw",
    absorb_transit: bool = False,
    absorb_neutral: bool = False,
    absorption_mode: str = "retrospective",
) -> dict:
    """
    Build Module-1 feature blocks and final dense matrices for GMM.

    Expected check-in columns:
      - SessionId
      - CheckinTime
      - PoiId
      - PoiCategoryName
      - Latitude
      - Longitude

    Parameters
    ----------
    taxonomy_level : str
        "raw", "mid" or "seg".
    absorb_transit : bool
        If True, transit check-ins inherit destination's category label.
    absorb_neutral : bool
        If True, neutral check-ins inherit preceding activity's label.
    absorption_mode : str
        "retrospective" (offline) or "online" (prefix, trailing transit unresolved).

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
    duration_mean, duration_std = fit_duration_normalizer_from_checkins(
        train_checkins,
        session_id_col="SessionId",
        checkin_time_col="CheckinTime",
        poi_id_col="PoiId",
    )

    temporal_kwargs = dict(
        session_id_col="SessionId",
        checkin_time_col="CheckinTime",
        poi_id_col="PoiId",
        duration_mean=duration_mean,
        duration_std=duration_std,
    )

    X_train_temp, train_temp_meta = build_temporal_feature_matrix(
        train_checkins,
        **temporal_kwargs,
    )
    X_val_temp, val_temp_meta = build_temporal_feature_matrix(
        val_checkins,
        **temporal_kwargs,
    )
    X_test_temp, test_temp_meta = build_temporal_feature_matrix(
        test_checkins,
        **temporal_kwargs,
    )

    cprint("Temporal features extracted successfully.", "green")

    # ------------------------------------------------------------------
    # 2) CATEGORY BLOCK
    # ------------------------------------------------------------------
    cprint(
        f"Extracting category features "
        f"(level={taxonomy_level}, absorb_transit={absorb_transit}, "
        f"absorb_neutral={absorb_neutral})...",
        "yellow",
    )

    cat_kwargs = dict(
        taxonomy_level=taxonomy_level,
        absorb_transit=absorb_transit,
        absorb_neutral=absorb_neutral,
        mode=absorption_mode,
        session_id_col="SessionId",
        checkin_time_col="CheckinTime",
        poi_id_col="PoiId",
        poi_category_name_col="PoiCategoryName",
    )

    train_cat_df = build_category_documents(train_checkins, **cat_kwargs)
    val_cat_df = build_category_documents(val_checkins, **cat_kwargs)
    test_cat_df = build_category_documents(test_checkins, **cat_kwargs)

    train_cat_df = align_session_dataframe(
        train_temp_meta, train_cat_df, name="train_cat_df"
    )
    val_cat_df = align_session_dataframe(val_temp_meta, val_cat_df, name="val_cat_df")
    test_cat_df = align_session_dataframe(
        test_temp_meta, test_cat_df, name="test_cat_df"
    )

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
        max_valid_components = min(
            X_train_cat_tfidf.shape[0] - 1, X_train_cat_tfidf.shape[1] - 1
        )
        if max_valid_components < 1:
            raise ValueError("Not enough training sessions or category vocabulary to run TruncatedSVD.")  # fmt: skip

        n_components = min(category_svd_components, max_valid_components)

        category_svd = TruncatedSVD(
            n_components=n_components, random_state=random_state
        )
        category_post_normalizer = Normalizer(norm="l2")

        # X_train_cat = category_post_normalizer.fit_transform(
        #     category_svd.fit_transform(X_train_cat_tfidf)
        # )
        # X_val_cat = category_post_normalizer.transform(
        #     category_svd.transform(X_val_cat_tfidf)
        # )
        # X_test_cat = category_post_normalizer.transform(
        #     category_svd.transform(X_test_cat_tfidf)
        # )
        X_train_cat = category_svd.fit_transform(X_train_cat_tfidf)
        X_val_cat = category_svd.transform(X_val_cat_tfidf)
        X_test_cat = category_svd.transform(X_test_cat_tfidf)

        explained_var = category_svd.explained_variance_ratio_.sum()
        cprint(
            f"  SVD: {n_components} components, "
            f"explained variance: {explained_var:.2%}",
            "cyan",
        )
    else:
        category_svd = None
        category_post_normalizer = None
        X_train_cat = X_train_cat_tfidf.toarray()
        X_val_cat = X_val_cat_tfidf.toarray()
        X_test_cat = X_test_cat_tfidf.toarray()

    vocab_size = len(vectorizer.vocabulary_)
    cprint(
        f"Category features extracted successfully. Vocabulary: {vocab_size} tokens.",
        "green",
    )

    X_train_cat = np.asarray(X_train_cat, dtype=np.float32)
    X_val_cat = np.asarray(X_val_cat, dtype=np.float32)
    X_test_cat = np.asarray(X_test_cat, dtype=np.float32)

    # ------------------------------------------------------------------
    # 3) SPATIAL BLOCK
    # ------------------------------------------------------------------
    cprint("Extracting spatial features...", "yellow")

    spatial_kwargs = dict(
        session_id_col="SessionId",
        checkin_time_col="CheckinTime",
        poi_id_col="PoiId",
        poi_latitude_col="Latitude",
        poi_longitude_col="Longitude",
        region_col=region_col,
        h3_resolution=h3_resolution,
    )

    train_spatial_df = build_session_spatial_aggregates(
        train_checkins, **spatial_kwargs
    )
    val_spatial_df = build_session_spatial_aggregates(val_checkins, **spatial_kwargs)
    test_spatial_df = build_session_spatial_aggregates(test_checkins, **spatial_kwargs)

    align_session_dataframe_kwargs = dict(
        name="train_spatial_df",
        session_id_col="SessionId",
    )

    train_spatial_df = align_session_dataframe(
        train_temp_meta, train_spatial_df, **align_session_dataframe_kwargs
    )
    val_spatial_df = align_session_dataframe(
        val_temp_meta, val_spatial_df, **align_session_dataframe_kwargs
    )
    test_spatial_df = align_session_dataframe(
        test_temp_meta, test_spatial_df, **align_session_dataframe_kwargs
    )

    spatial_cols = [
        "movement_radius_km",
        "h3_entropy",
        "start_end_centroid_displacement_km",
    ]

    spatial_scaler = StandardScaler()
    X_train_spatial = spatial_scaler.fit_transform(
        train_spatial_df[spatial_cols].to_numpy(dtype=np.float32)
    )
    X_val_spatial = spatial_scaler.transform(
        val_spatial_df[spatial_cols].to_numpy(dtype=np.float32)
    )
    X_test_spatial = spatial_scaler.transform(
        test_spatial_df[spatial_cols].to_numpy(dtype=np.float32)
    )

    X_train_spatial = np.asarray(X_train_spatial, dtype=np.float32)
    X_val_spatial = np.asarray(X_val_spatial, dtype=np.float32)
    X_test_spatial = np.asarray(X_test_spatial, dtype=np.float32)

    cprint("Spatial features extracted successfully.", "green")

    # ------------------------------------------------------------------
    # 4) FINAL DENSE MATRICES FOR GMM
    # ------------------------------------------------------------------
    temporal_scaler = StandardScaler()
    X_train_temp = temporal_scaler.fit_transform(X_train_temp)
    X_val_temp = temporal_scaler.transform(X_val_temp)
    X_test_temp = temporal_scaler.transform(X_test_temp)

    category_scaler = StandardScaler()
    X_train_cat = category_scaler.fit_transform(X_train_cat)
    X_val_cat = category_scaler.transform(X_val_cat)
    X_test_cat = category_scaler.transform(X_test_cat)

    X_train_temp = np.asarray(X_train_temp, dtype=np.float32)
    X_val_temp = np.asarray(X_val_temp, dtype=np.float32)
    X_test_temp = np.asarray(X_test_temp, dtype=np.float32)
    X_train_cat = np.asarray(X_train_cat, dtype=np.float32)
    X_val_cat = np.asarray(X_val_cat, dtype=np.float32)
    X_test_cat = np.asarray(X_test_cat, dtype=np.float32)

    # category_weight = 0.5

    X_train = np.hstack([X_train_temp, X_train_cat, X_train_spatial]).astype(np.float32)  # fmt: skip
    X_val = np.hstack([X_val_temp, X_val_cat, X_val_spatial]).astype(np.float32)  # fmt: skip
    X_test = np.hstack([X_test_temp, X_test_cat, X_test_spatial]).astype(np.float32)  # fmt:skip
    # X_train = np.hstack([X_train_cat]).astype(np.float32)
    # X_val = np.hstack([X_val_cat]).astype(np.float32)
    # X_test = np.hstack([X_test_cat]).astype(np.float32)

    cprint(
        f"Final feature matrix: {X_train.shape[1]} dims "
        f"(temporal={X_train_temp.shape[1]}, "
        f"category={X_train_cat.shape[1]}, "
        f"spatial={X_train_spatial.shape[1]})",
        "cyan",
    )

    # ------------------------------------------------------------------
    # 5) METADATA
    # ------------------------------------------------------------------
    meta_cat_cols = ["SessionId", "category_sequence", "category_doc"]
    if "category_sequence_raw" in train_cat_df.columns:
        meta_cat_cols.append("category_sequence_raw")

    meta_cols = ["SessionId", "session_start_time", "session_end_time"]
    train_meta = (
        train_temp_meta[meta_cols]
        .merge(train_cat_df[meta_cat_cols], on="SessionId", how="left")
        .merge(
            train_spatial_df[["SessionId"] + spatial_cols], on="SessionId", how="left"
        )
    )

    val_meta = (
        val_temp_meta[meta_cols]
        .merge(val_cat_df[meta_cat_cols], on="SessionId", how="left")
        .merge(val_spatial_df[["SessionId"] + spatial_cols], on="SessionId", how="left")
    )

    test_meta = (
        test_temp_meta[meta_cols]
        .merge(test_cat_df[meta_cat_cols], on="SessionId", how="left")
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
            "temporal_scaler": temporal_scaler,
            "category_scaler": category_scaler,
            "spatial_scaler": spatial_scaler,
            "spatial_cols": spatial_cols,
            "taxonomy_level": taxonomy_level,
            "absorb_transit": absorb_transit,
            "absorb_neutral": absorb_neutral,
        },
    }
