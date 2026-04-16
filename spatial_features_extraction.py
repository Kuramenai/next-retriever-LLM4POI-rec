import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Great-circle distance in kilometers.
    Works with scalars or numpy arrays.
    """
    R = 6371.0088

    lat1 = np.radians(np.asarray(lat1, dtype=np.float64))
    lon1 = np.radians(np.asarray(lon1, dtype=np.float64))
    lat2 = np.radians(np.asarray(lat2, dtype=np.float64))
    lon2 = np.radians(np.asarray(lon2, dtype=np.float64))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c


def radius_of_gyration_km(lats: np.ndarray, lons: np.ndarray) -> float:
    """
    Session spread around its centroid.
    This is a principled definition of 'movement radius'.
    """
    centroid_lat = float(np.mean(lats))
    centroid_lon = float(np.mean(lons))

    dists = haversine_km(lats, lons, centroid_lat, centroid_lon)
    rog = float(np.sqrt(np.mean(np.square(dists))))
    return rog


def shannon_entropy(tokens, normalize: bool = True) -> float:
    """
    Entropy over discrete region tokens.
    If normalize=True, returns value in [0, 1] approximately.
    """
    if len(tokens) == 0:
        return 0.0

    counts = pd.Series(tokens).value_counts().to_numpy(dtype=np.float64)
    probs = counts / counts.sum()

    ent = float(-(probs * np.log(probs + 1e-12)).sum())

    if normalize and len(counts) > 1:
        ent /= float(np.log(len(counts)))

    return ent


def h3_cell_from_latlon(lat: float, lon: float, resolution: int = 7) -> str:
    """
    Compute H3 cell token. Supports both newer and older h3-py APIs.
    """
    try:
        import h3
    except ImportError as e:
        raise ImportError(
            "h3 is not installed. Install it with `pip install h3`, "
            "or precompute a region_token column and pass region_col='region_token'."
        ) from e

    if hasattr(h3, "latlng_to_cell"):  # h3-py v4+
        return h3.latlng_to_cell(lat, lon, resolution)
    if hasattr(h3, "geo_to_h3"):  # older API
        return h3.geo_to_h3(lat, lon, resolution)

    raise RuntimeError("Unsupported h3 package version.")


def centroid_displacement_km(lats: np.ndarray, lons: np.ndarray) -> float:
    """
    Distance between centroid of early half and centroid of late half.
    This is a robust version of session start-end spatial drift.
    """
    n = len(lats)
    if n <= 1:
        return 0.0

    split = max(1, n // 2)

    early_lats = lats[:split]
    early_lons = lons[:split]
    late_lats = lats[split:]
    late_lons = lons[split:]

    if len(late_lats) == 0:
        return 0.0

    early_centroid_lat = float(np.mean(early_lats))
    early_centroid_lon = float(np.mean(early_lons))
    late_centroid_lat = float(np.mean(late_lats))
    late_centroid_lon = float(np.mean(late_lons))

    return float(
        haversine_km(
            early_centroid_lat,
            early_centroid_lon,
            late_centroid_lat,
            late_centroid_lon,
        )
    )


def build_session_spatial_aggregates(
    session_checkins_df: pd.DataFrame,
    region_col: str | None = None,
    h3_resolution: int = 7,
    normalize_entropy: bool = True,
) -> pd.DataFrame:
    """
    Build the Module-1 coarse spatial feature block per session.

    Required columns:
      - SessionId
      - Time
      - Latitude
      - Longitude

    Optional:
      - region_col: precomputed region token column, e.g. 'region_token'
                    If absent, H3 will be computed on the fly.

    Returns one row per session with:
      - movement_radius_km
      - h3_entropy
      - start_end_centroid_displacement_km
    """
    required = {"SessionId", "Time", "Latitude", "Longitude"}
    missing = required - set(session_checkins_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = session_checkins_df.copy()
    df["Time"] = pd.to_datetime(df["Time"], errors="raise")

    sort_cols = ["SessionId", "Time"]
    if "PId" in df.columns:
        sort_cols.append("PId")

    df = df.sort_values(sort_cols).reset_index(drop=True)

    # region token source:
    if region_col is not None:
        if region_col not in df.columns:
            raise ValueError(f"region_col='{region_col}' not found in dataframe")
        df["_region_token"] = df[region_col].astype(str)
    else:
        df["_region_token"] = [
            h3_cell_from_latlon(lat, lon, resolution=h3_resolution)
            for lat, lon in zip(df["Latitude"], df["Longitude"])
        ]

    rows = []

    for session_id, group in df.groupby("SessionId", sort=False):
        g = group.sort_values(sort_cols[1:]).reset_index(drop=True)

        lats = g["Latitude"].astype(float).to_numpy()
        lons = g["Longitude"].astype(float).to_numpy()
        region_tokens = g["_region_token"].tolist()

        movement_radius = radius_of_gyration_km(lats, lons)
        h3_ent = shannon_entropy(region_tokens, normalize=normalize_entropy)
        displacement = centroid_displacement_km(lats, lons)

        rows.append(
            {
                "SessionId": session_id,
                "movement_radius_km": movement_radius,
                "h3_entropy": h3_ent,
                "start_end_centroid_displacement_km": displacement,
            }
        )

    return pd.DataFrame(rows)
