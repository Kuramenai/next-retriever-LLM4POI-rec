import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture
from termcolor import cprint
from tqdm import tqdm
from collections import Counter


import hdbscan

try:
    # When imported as a package module.
    from .features_extraction import build_feature_blocks  # type: ignore
except Exception:
    # When executed as a script from within this folder.
    from features_extraction import build_feature_blocks

pd.set_option("future.no_silent_downcasting", True)


def suggest_k_hdbscan(
    X: np.ndarray,
    *,
    min_cluster_sizes: tuple[int, ...] | None = None,
    min_samples_values: tuple[int | None, ...] = (None,),
    metric: str = "euclidean",
) -> dict:
    """
    Use HDBSCAN as a diagnostic to suggest a plausible number of clusters (K).

    Notes
    -----
    - HDBSCAN can label points as noise (-1). We report cluster counts excluding noise.
    - The number of clusters is sensitive to hyperparameters; treat output as a guide.

    Returns
    -------
    dict with:
      - suggested_k (int | None)
      - suggested_k_range (tuple[int, int] | None)
      - suggested_candidate_K (tuple[int, ...] | None)
      - diagnostics (pd.DataFrame)
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    n = int(X.shape[0])
    if n < 5:
        return {
            "suggested_k": None,
            "suggested_k_range": None,
            "suggested_candidate_K": None,
            "diagnostics": pd.DataFrame(
                [{"n_samples": n, "error": "Too few samples for HDBSCAN diagnostic"}]
            ),
        }

    if min_cluster_sizes is None:
        # Heuristic defaults that scale with dataset size.
        # Keep small values too, otherwise HDBSCAN may return 0 clusters on small sets.
        # min_cluster_sizes = tuple(
        #     sorted(
        #         {
        #             max(5, int(0.01 * n)),
        #             max(8, int(0.02 * n)),
        #             max(12, int(0.03 * n)),
        #             max(20, int(0.05 * n)),
        #         }
        #     )
        # )
        min_cluster_sizes = (10, 20, 50, 100, 200, 300)
        min_samples_values = (5, 10, 15)

    rows: list[dict] = []
    for mcs in tqdm(min_cluster_sizes, desc="Testing min_cluster_sizes"):
        for ms in min_samples_values:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=int(mcs),
                min_samples=ms,
                metric=metric,
                prediction_data=False,
            )
            labels = clusterer.fit_predict(X)
            labels = np.asarray(labels, dtype=int)

            noise = int(np.sum(labels == -1))
            noise_frac = float(noise / n)
            unique = sorted(set(labels.tolist()))
            clusters = [c for c in unique if c != -1]
            k = int(len(clusters))
            rows.append(
                {
                    "min_cluster_size": int(mcs),
                    "min_samples": None if ms is None else int(ms),
                    "metric": str(metric),
                    "n_samples": n,
                    "n_clusters_excluding_noise": k,
                    "noise_frac": noise_frac,
                }
            )

    diagnostics = pd.DataFrame(rows).sort_values(
        by=["n_clusters_excluding_noise", "noise_frac", "min_cluster_size"],
        ascending=[False, True, True],
    )

    ks = [
        int(v) for v in diagnostics["n_clusters_excluding_noise"].tolist() if int(v) > 0
    ]
    if not ks:
        return {
            "suggested_k": None,
            "suggested_k_range": None,
            "suggested_candidate_K": None,
            "diagnostics": diagnostics,
        }

    # Robust suggested K: mode over the grid; break ties by choosing the smaller K.
    counts = Counter(ks)
    max_count = max(counts.values())
    modes = sorted([k for k, c in counts.items() if c == max_count])
    suggested_k = int(modes[0])

    k_min = int(min(ks))
    k_max = int(max(ks))
    suggested_k_range = (k_min, k_max)

    # Build a compact candidate_K tuple around the suggested_k.
    # (Clip to >=2; include a small spread for BIC search.)
    cand = sorted(
        {max(2, suggested_k + d) for d in (-10, -5, -2, 0, 2, 5, 10)}
        | {max(2, k_min), max(2, k_max)}
    )

    return {
        "suggested_k": suggested_k,
        "suggested_k_range": suggested_k_range,
        "suggested_candidate_K": tuple(int(x) for x in cand),
        "diagnostics": diagnostics,
    }


def _top_m_from_proba(proba: np.ndarray, top_m: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return top-M component ids and their probabilities for each row.
    """
    top_idx = np.argsort(-proba, axis=1)[:, :top_m]
    top_scores = np.take_along_axis(proba, top_idx, axis=1)
    return top_idx, top_scores


def _build_assignment_table(
    proba: np.ndarray,
    meta: pd.DataFrame | None = None,
    top_m: int = 3,
) -> pd.DataFrame:
    """
    Build a session-level routing table from posterior probabilities.
    """
    hard_labels = proba.argmax(axis=1)
    max_probs = proba.max(axis=1)

    sorted_probs = np.sort(proba, axis=1)[:, ::-1]
    top1 = sorted_probs[:, 0]
    top2 = (
        sorted_probs[:, 1]
        if proba.shape[1] > 1
        else np.zeros(len(proba), dtype=np.float32)
    )
    top2_gap = top1 - top2

    top_ids, top_scores = _top_m_from_proba(proba, top_m=top_m)

    df = meta.copy() if meta is not None else pd.DataFrame(index=np.arange(len(proba)))
    df = df.reset_index(drop=True)

    df["prototype_id"] = hard_labels.astype(int)
    df["prototype_confidence"] = max_probs.astype(np.float32)
    df["prototype_top2_gap"] = top2_gap.astype(np.float32)

    m = int(min(top_m, proba.shape[1]))
    for j in range(m):
        df[f"top{j + 1}_prototype_id"] = top_ids[:, j].astype(int)
        df[f"top{j + 1}_prototype_prob"] = top_scores[:, j].astype(np.float32)

    return df


def fit_gmm_prototypes(
    X_train: np.ndarray,
    train_meta: pd.DataFrame | None = None,
    X_val: np.ndarray | None = None,
    val_meta: pd.DataFrame | None = None,
    X_test: np.ndarray | None = None,
    test_meta: pd.DataFrame | None = None,
    *,
    candidate_K: tuple[int, ...] = (10, 20, 30, 50),
    candidate_covariance_types: tuple[str, ...] = ("diag", "full"),
    top_m: int = 3,
    n_init: int = 5,
    max_iter: int = 300,
    reg_covar: float = 1e-6,
    random_state: int = 42,
) -> dict:
    """
    Fit GMM prototypes with BIC-based model selection on training data.

    Parameters
    ----------
    X_train, X_val, X_test
        Dense feature matrices from build_feature_blocks(...).
    train_meta, val_meta, test_meta
        Optional session metadata aligned row-wise with X_*.
    candidate_K
        Candidate numbers of mixture components.
    candidate_covariance_types
        Candidate covariance types for sklearn GaussianMixture.
        'diag' is usually safer in moderate/high dimensions.
    top_m
        Number of top routing components to store per session.

    Returns
    -------
    dict containing:
      - best fitted model
      - model selection table
      - posterior probabilities
      - assignment / routing tables
      - prototype summary table
    """
    if X_train.ndim != 2:
        raise ValueError("X_train must be a 2D array")

    model_rows = []
    best_model = None
    best_config = None
    best_bic = np.inf
    best_converged_model = None
    best_converged_config = None
    best_converged_bic = np.inf

    for covariance_type in tqdm(
        candidate_covariance_types,
        dynamic_ncols=True,
        total=len(candidate_covariance_types),
        desc="Finding best covariance type",
    ):
        for K in tqdm(
            candidate_K,
            dynamic_ncols=True,
            total=len(candidate_K),
            desc="Finding best K",
        ):
            try:
                gmm = GaussianMixture(
                    n_components=K,
                    covariance_type=covariance_type,
                    n_init=n_init,
                    max_iter=max_iter,
                    reg_covar=reg_covar,
                    init_params="kmeans",
                    random_state=random_state,
                )
                gmm.fit(X_train)

                bic = gmm.bic(X_train)
                aic = gmm.aic(X_train)
                train_avg_loglik = gmm.score(X_train)
                val_avg_loglik = gmm.score(X_val) if X_val is not None else np.nan

                model_rows.append(
                    {
                        "K": K,
                        "covariance_type": covariance_type,
                        "bic": float(bic),
                        "aic": float(aic),
                        "train_avg_loglik": float(train_avg_loglik),
                        "val_avg_loglik": float(val_avg_loglik)
                        if not np.isnan(val_avg_loglik)
                        else np.nan,
                        "converged": bool(gmm.converged_),
                        "n_iter": int(gmm.n_iter_),
                    }
                )

                if bic < best_bic:
                    best_bic = bic
                    best_model = gmm
                    best_config = {
                        "K": K,
                        "covariance_type": covariance_type,
                        "bic": float(bic),
                        "aic": float(aic),
                    }
                if bool(gmm.converged_) and bic < best_converged_bic:
                    best_converged_bic = bic
                    best_converged_model = gmm
                    best_converged_config = {
                        "K": K,
                        "covariance_type": covariance_type,
                        "bic": float(bic),
                        "aic": float(aic),
                    }

            except Exception as e:
                model_rows.append(
                    {
                        "K": K,
                        "covariance_type": covariance_type,
                        "bic": np.nan,
                        "aic": np.nan,
                        "train_avg_loglik": np.nan,
                        "val_avg_loglik": np.nan,
                        "converged": False,
                        "n_iter": np.nan,
                        "error": repr(e),
                    }
                )

    # Prefer a converged solution when available; fall back to best BIC overall.
    if best_converged_model is not None:
        best_model = best_converged_model
        best_config = best_converged_config

    if best_model is None:
        raise RuntimeError("All GMM fits failed. Check feature matrix scale or reduce model complexity.")  # fmt: skip

    model_selection_df = (
        pd.DataFrame(model_rows)
        .sort_values(by=["bic", "K"], ascending=[True, True], na_position="last")
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # Posterior probabilities
    # ------------------------------------------------------------
    train_proba = best_model.predict_proba(X_train)
    val_proba = best_model.predict_proba(X_val) if X_val is not None else None
    test_proba = best_model.predict_proba(X_test) if X_test is not None else None

    train_assignments = _build_assignment_table(
        train_proba, meta=train_meta, top_m=top_m
    )
    val_assignments = (
        _build_assignment_table(val_proba, meta=val_meta, top_m=top_m)
        if val_proba is not None
        else None
    )
    test_assignments = (
        _build_assignment_table(test_proba, meta=test_meta, top_m=top_m)
        if test_proba is not None
        else None
    )

    # ------------------------------------------------------------
    # Prototype summary
    # ------------------------------------------------------------
    hard_train_labels = train_proba.argmax(axis=1)
    # hard_counts = pd.Series(hard_train_labels).value_counts().sort_index()

    prototype_rows = []
    for k in range(best_model.n_components):
        component_mask = hard_train_labels == k
        component_size = int(component_mask.sum())
        mean_posterior = (
            float(train_proba[component_mask, k].mean()) if component_size > 0 else 0.0
        )

        prototype_rows.append(
            {
                "prototype_id": int(k),
                "mixture_weight": float(best_model.weights_[k]),
                "train_hard_count": component_size,
                "train_hard_fraction": float(component_size / len(X_train)),
                "mean_self_posterior": mean_posterior,
            }
        )

    prototype_summary_df = (
        pd.DataFrame(prototype_rows)
        .sort_values(by="mixture_weight", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "model": best_model,
        "best_config": best_config,
        "model_selection": model_selection_df,
        "train": {
            "proba": train_proba,
            "assignments": train_assignments,
        },
        "val": None
        if val_proba is None
        else {
            "proba": val_proba,
            "assignments": val_assignments,
        },
        "test": None
        if test_proba is None
        else {
            "proba": test_proba,
            "assignments": test_assignments,
        },
        "prototype_summary": prototype_summary_df,
    }


if __name__ == "__main__":
    cprint("Starting GMM prototypes fitting...", "yellow")
    cprint("Loading check-in data...", "yellow")

    city = "nyc"
    run_hdbscan_k_diagnostic = False
    scrip_dir = Path(__file__).resolve().parent.parent
    out_dir = scrip_dir / f"data/{city}"
    train_checkins = pd.read_csv(out_dir / "train_sample.csv")
    val_checkins = pd.read_csv(out_dir / "validate_sample_with_traj.csv")
    test_checkins = pd.read_csv(out_dir / "test_sample.csv")

    session_id_col_mapping = {
        "pseudo_session_trajectory_id": "SessionId",
    }
    train_checkins = train_checkins.rename(columns=session_id_col_mapping)
    train_checkins["CheckinTime"] = pd.to_datetime(train_checkins["UTCTimeOffset"])

    val_checkins = val_checkins.rename(columns=session_id_col_mapping)
    val_checkins["CheckinTime"] = pd.to_datetime(val_checkins["UTCTimeOffset"])

    test_checkins = test_checkins.rename(columns=session_id_col_mapping)
    test_checkins["CheckinTime"] = pd.to_datetime(test_checkins["UTCTimeOffset"])

    cprint("Check-in data loaded successfully.", "green")

    cprint("Building feature blocks...", "yellow")
    feature_data = build_feature_blocks(
        train_checkins=train_checkins,
        val_checkins=val_checkins,
        test_checkins=test_checkins,
        taxonomy_level="raw",
        absorb_transit=False,
        absorb_neutral=False,
        category_svd_components=32,
    )

    cprint("Feature blocks built successfully.", "green")

    if run_hdbscan_k_diagnostic:
        cprint("HDBSCAN diagnostic: suggesting K...", "yellow")
        try:
            k_diag = suggest_k_hdbscan(feature_data["train"]["X"])
            cprint(
                f"  suggested_k={k_diag['suggested_k']}, "
                f"suggested_k_range={k_diag['suggested_k_range']}, "
                f"suggested_candidate_K={k_diag['suggested_candidate_K']}",
                "cyan",
            )
            diag_path = scrip_dir / f"artifacts/{city}/{city}_hdbscan_k_diagnostic.csv"
            diag_path.parent.mkdir(parents=True, exist_ok=True)
            k_diag["diagnostics"].to_csv(diag_path, index=False)
            cprint(f"  wrote diagnostics to {diag_path}", "green")
        except Exception as e:
            cprint(f"HDBSCAN diagnostic failed: {e!r}", "red")

    cprint("Fitting GMM prototypes...", "yellow")
    gmm_data = fit_gmm_prototypes(
        X_train=feature_data["train"]["X"],
        train_meta=feature_data["train"]["meta"],
        X_val=feature_data["val"]["X"],
        val_meta=feature_data["val"]["meta"],
        X_test=feature_data["test"]["X"],
        test_meta=feature_data["test"]["meta"],
        candidate_K=(8, 10, 12, 15, 20),
        candidate_covariance_types=("spherical", "diag", "tied"),
        reg_covar=1e-4,
        top_m=3,
    )

    cprint("GMM prototypes fitted successfully.", "green")

    cprint("Best configuration: ", "yellow", end="")
    print(gmm_data["best_config"])
    cprint("Model selection table: ", "yellow", end="")
    print(gmm_data["model_selection"].head())
    cprint("Prototype summary: ", "yellow", end="")
    print(gmm_data["prototype_summary"].head())

    cprint("Saving GMM data", "yellow")
    gmm_path = scrip_dir / f"artifacts/{city}/{city}_gmm_cluster.pkl"
    gmm_path.parent.mkdir(parents=True, exist_ok=True)
    with gmm_path.open("wb") as f:
        pickle.dump(gmm_data, f)

    cprint("Saving features", "yellow")
    features_path = scrip_dir / f"artifacts/{city}/{city}_features.pkl"
    features_path.parent.mkdir(parents=True, exist_ok=True)
    with features_path.open("wb") as f:
        pickle.dump(feature_data, f)
