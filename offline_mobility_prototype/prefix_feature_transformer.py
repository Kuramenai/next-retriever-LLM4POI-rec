from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize

from offline_mobility_prototype.temporal_features_extraction import (
    extract_temporal_features,
)
from offline_mobility_prototype.categorical_features_extraction import (
    build_category_documents,
)
from offline_mobility_prototype.spatial_features_extraction import (
    build_session_spatial_aggregates,
)


def _first_present(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _require_single_session(prefix_df: pd.DataFrame, session_id_col: str) -> Any:
    if session_id_col not in prefix_df.columns:
        return "__PREFIX_SESSION__"
    vals = prefix_df[session_id_col].dropna().unique().tolist()
    if len(vals) == 0:
        return "__PREFIX_SESSION__"
    if len(vals) > 1:
        raise ValueError(
            f"prefix_df contains multiple session ids in {session_id_col!r}: {vals[:5]}"
        )
    return vals[0]


@dataclass
class FrozenModule1Artifacts:
    """
    Frozen preprocessing artifacts created during Module-1 training.

    This mirrors `build_feature_blocks(...)[\"artifacts\"]` from
    `offline_mobility_prototype/features_extraction.py`.
    """

    duration_mean: float
    duration_std: float

    category_vectorizer: object
    category_svd: Optional[object]
    category_post_normalizer: Optional[object]

    spatial_scaler: object
    spatial_cols: list[str]


class FrozenModule1PrefixTransformer:
    """
    Transform a single prefix/session check-ins DataFrame into the same
    feature schema used to fit the Module-1 GMM.

    It must NOT refit anything; it only reuses frozen artifacts from training.

    Output:
      - a single-row pandas DataFrame (1 × D) with deterministic column names
      - `feature_cols` property listing the columns in order

    Notes on expected prefix_df schema
    ---------------------------------
    Training code (Module-1) expects session-level check-ins with columns:
      - SessionId, Time, PId, Category, Latitude, Longitude

    This transformer is defensive and can map common alternatives via the
    constructor’s `*_col` parameters (e.g., UTCTimeOffset, PoiId, etc.).
    """

    def __init__(
        self,
        artifacts: FrozenModule1Artifacts,
        *,
        session_id_col: str = "SessionId",
        timestamp_col: str = "Time",
        poi_id_col: str = "PId",
        category_col: str = "Category",
        lat_col: str = "Latitude",
        lon_col: str = "Longitude",
        region_col: str | None = None,
        h3_resolution: int = 7,
        category_ngram_range: tuple[int, int] = (1, 2),
    ):
        self.artifacts = artifacts
        self.session_id_col = session_id_col
        self.timestamp_col = timestamp_col
        self.poi_id_col = poi_id_col
        self.category_col = category_col
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.region_col = region_col
        self.h3_resolution = int(h3_resolution)

        # Only used for validation / sanity; the actual vectorizer is frozen.
        self.category_ngram_range = category_ngram_range

        self._feature_cols: list[str] = self._build_feature_cols()

    @classmethod
    def from_feature_blocks_output(
        cls,
        feature_blocks_out: dict,
        *,
        region_col: str | None = None,
        h3_resolution: int = 8,
        session_id_col: str = "SessionId",
        timestamp_col: str = "Time",
        poi_id_col: str = "PId",
        category_col: str = "Category",
        lat_col: str = "Latitude",
        lon_col: str = "Longitude",
    ) -> "FrozenModule1PrefixTransformer":
        """
        Convenience constructor using the output of build_feature_blocks(...).
        """
        a = feature_blocks_out["artifacts"]
        artifacts = FrozenModule1Artifacts(
            duration_mean=float(a["duration_mean"]),
            duration_std=float(a["duration_std"]),
            category_vectorizer=a["category_vectorizer"],
            category_svd=a.get("category_svd", None),
            category_post_normalizer=a.get("category_post_normalizer", None),
            spatial_scaler=a["spatial_scaler"],
            spatial_cols=list(a["spatial_cols"]),
        )
        return cls(
            artifacts,
            region_col=region_col,
            h3_resolution=h3_resolution,
            session_id_col=session_id_col,
            timestamp_col=timestamp_col,
            poi_id_col=poi_id_col,
            category_col=category_col,
            lat_col=lat_col,
            lon_col=lon_col,
        )

    @property
    def feature_cols(self) -> list[str]:
        return list(self._feature_cols)

    def _build_feature_cols(self) -> list[str]:
        # Temporal block shape is fixed by extract_temporal_features()
        temporal_cols = [
            "temp_tod_morning_prop",
            "temp_tod_afternoon_prop",
            "temp_tod_evening_prop",
            "temp_tod_late_night_prop",
            "temp_start_is_weekend",
            "temp_duration_feature",
        ]

        # Category block dimensionality depends on whether SVD compression is used.
        if self.artifacts.category_svd is not None:
            n_cat = int(getattr(self.artifacts.category_svd, "n_components", 0))
            if n_cat <= 0:
                raise ValueError("category_svd has invalid n_components.")
            category_cols = [f"cat_svd_{i}" for i in range(n_cat)]
        else:
            # TF-IDF dimensionality depends on the fitted vocabulary size.
            vocab = getattr(self.artifacts.category_vectorizer, "vocabulary_", None)
            if vocab is None:
                raise ValueError("category_vectorizer is missing vocabulary_.")
            n_cat = int(len(vocab))
            category_cols = [f"cat_tfidf_{i}" for i in range(n_cat)]

        spatial_cols = [f"sp_{c}" for c in self.artifacts.spatial_cols]
        return [*temporal_cols, *category_cols, *spatial_cols]

    def _coerce_to_training_schema(self, prefix_df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(prefix_df, pd.DataFrame):
            raise TypeError("prefix_df must be a pandas DataFrame.")
        if prefix_df.empty:
            raise ValueError("prefix_df is empty.")

        df = prefix_df.copy()

        # Resolve likely source columns
        src_session = _first_present(df, [self.session_id_col, "SessionId", "session_id"])
        src_time = _first_present(
            df,
            [
                self.timestamp_col,
                "Time",
                "UTCTimeOffset",
                "timestamp",
                "Datetime",
                "date_time",
            ],
        )
        src_poi = _first_present(df, [self.poi_id_col, "PId", "PoiId", "POIId", "poi_id"])
        src_cat = _first_present(df, [self.category_col, "Category", "PoiCategoryName", "category"])
        src_lat = _first_present(df, [self.lat_col, "Latitude", "lat"])
        src_lon = _first_present(df, [self.lon_col, "Longitude", "lon", "lng"])

        missing = []
        if src_time is None:
            missing.append("timestamp")
        if src_poi is None:
            missing.append("poi_id")
        if src_cat is None:
            missing.append("category")
        if src_lat is None:
            missing.append("latitude")
        if src_lon is None:
            missing.append("longitude")
        if missing:
            raise ValueError(
                "prefix_df is missing required columns to build Module-1 features: "
                + ", ".join(missing)
            )

        session_id = _require_single_session(df, src_session or self.session_id_col)

        out = pd.DataFrame(
            {
                "SessionId": session_id,
                "Time": pd.to_datetime(df[src_time], errors="coerce"),
                "PId": df[src_poi],
                "Category": df[src_cat],
                "Latitude": pd.to_numeric(df[src_lat], errors="coerce"),
                "Longitude": pd.to_numeric(df[src_lon], errors="coerce"),
            }
        )

        if out["Time"].isna().any():
            bad = int(out["Time"].isna().sum())
            raise ValueError(f"{bad} rows have invalid timestamps after parsing.")

        # If region_col is requested, pass it through if present.
        if self.region_col is not None:
            if self.region_col in df.columns:
                out[self.region_col] = df[self.region_col].astype(str)

        return out

    def transform_prefix(self, prefix_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a prefix session into a single-row feature DataFrame.
        """
        df = self._coerce_to_training_schema(prefix_df)

        # ---- Temporal block (6D) ----
        temp = extract_temporal_features(
            df,
            duration_mean=self.artifacts.duration_mean,
            duration_std=self.artifacts.duration_std,
        )
        x_temp = np.asarray(temp["vector"], dtype=np.float32).reshape(1, -1)

        # ---- Category block ----
        cat_df = build_category_documents(df)
        if len(cat_df) != 1:
            raise RuntimeError(
                f"Expected one category document for the prefix but got {len(cat_df)}."
            )

        doc = cat_df["category_doc"].iloc[0]
        x_tfidf = self.artifacts.category_vectorizer.transform([doc])
        x_tfidf = normalize(x_tfidf, norm="l2")

        if self.artifacts.category_svd is not None:
            x_cat = self.artifacts.category_svd.transform(x_tfidf)
            if self.artifacts.category_post_normalizer is not None:
                x_cat = self.artifacts.category_post_normalizer.transform(x_cat)
            x_cat = np.asarray(x_cat, dtype=np.float32)
        else:
            x_cat = np.asarray(x_tfidf.toarray(), dtype=np.float32)

        # ---- Spatial block (scaled) ----
        sp_df = build_session_spatial_aggregates(
            df,
            region_col=self.region_col,
            h3_resolution=self.h3_resolution,
        )
        sp_df = sp_df[["SessionId", *self.artifacts.spatial_cols]].copy()
        x_sp = self.artifacts.spatial_scaler.transform(
            sp_df[self.artifacts.spatial_cols].to_numpy(dtype=np.float32)
        )
        x_sp = np.asarray(x_sp, dtype=np.float32)

        # ---- Final dense vector ----
        x = np.hstack([x_temp, x_cat, x_sp]).astype(np.float32)

        if x.shape[0] != 1:
            raise RuntimeError("transform_prefix() must return exactly one row.")

        if x.shape[1] != len(self._feature_cols):
            raise RuntimeError(
                f"Feature dimension mismatch: got {x.shape[1]} but expected {len(self._feature_cols)}."
            )

        return pd.DataFrame(x, columns=self._feature_cols)

