import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture
from features_extraction import build_feature_blocks
from termcolor import cprint
from tqdm import tqdm


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

    for j in range(top_m):
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

    for covariance_type in tqdm(
        candidate_covariance_types,
        dynamic_ncols=True,
        total=len(candidate_covariance_types),
        desc="Finding best covariance type",
    ):
        for K in candidate_K:
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

    if best_model is None:
        raise RuntimeError(
            "All GMM fits failed. Check feature matrix scale or reduce model complexity."
        )

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
    hard_counts = pd.Series(hard_train_labels).value_counts().sort_index()

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


# fmt: off
if __name__ == "__main__":
    cprint("Starting GMM prototypes fitting...", "yellow")
    cprint("Loading check-in data...", "yellow")

    city = "nyc"
    out_dir = Path(f"data/{city}")
    train_checkins = pd.read_csv(out_dir / "train_sample.csv")
    val_checkins = pd.read_csv(out_dir / "validate_sample_with_traj.csv")
    test_checkins = pd.read_csv(out_dir / "test_sample.csv")

    train_checkins = train_checkins.rename(
        columns={"pseudo_session_trajectory_id": "SessionId", "PoiCategoryId": "PId", "PoiCategoryName": "Category"})
    train_checkins["Time"] = pd.to_datetime(train_checkins["UTCTimeOffset"])

    val_checkins = val_checkins.rename(
        columns={"pseudo_session_trajectory_id": "SessionId", "PoiCategoryId": "PId", "PoiCategoryName": "Category"})
    val_checkins["Time"] = pd.to_datetime(val_checkins["UTCTimeOffset"])

    test_checkins = test_checkins.rename(
        columns={"pseudo_session_trajectory_id": "SessionId", "PoiCategoryId": "PId", "PoiCategoryName": "Category"})
    test_checkins["Time"] = pd.to_datetime(test_checkins["UTCTimeOffset"])

    cprint("Check-in data loaded successfully.", "green")

    cprint("Building feature blocks...", "yellow")
    feature_data = build_feature_blocks(
        train_checkins=train_checkins,
        val_checkins=val_checkins,
        test_checkins=test_checkins,
    )

    cprint("Feature blocks built successfully.", "green")

    cprint("Fitting GMM prototypes...", "yellow")
    gmm_data = fit_gmm_prototypes(
        X_train=feature_data["train"]["X"],
        train_meta=feature_data["train"]["meta"],
        X_val=feature_data["val"]["X"],
        val_meta=feature_data["val"]["meta"],
        X_test=feature_data["test"]["X"],
        test_meta=feature_data["test"]["meta"]
    )

    cprint("GMM prototypes fitted successfully.", "green")

    cprint("Best configuration: ", "yellow", end="")
    print(gmm_data["best_config"])
    cprint("Model selection table: ", "yellow", end="")
    print(gmm_data["model_selection"].head())
    cprint("Prototype summary: ", "yellow", end="")
    print(gmm_data["prototype_summary"].head())

    cprint("Saving GMM data", "yellow")
    gmm_path = Path(f"artifacts/{city}/{city}_gmm_cluster.pkl")
    gmm_path.parent.mkdir(parents=True, exist_ok=True)
    with gmm_path.open("wb") as f:
        pickle.dump(gmm_data, f)
    
    cprint("Saving features", "yellow")
    features_path = Path(f"artifacts/{city}/{city}_features.pkl")
    features_path.parent.mkdir(parents=True, exist_ok=True)
    with features_path.open("wb") as f:
        pickle.dump(feature_data, f)
        
