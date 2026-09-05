from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import shap


def _sample_rows(frame: pd.DataFrame, max_samples: int, random_state: int) -> pd.DataFrame:
    """Return a deterministic row sample while preserving DataFrame columns."""
    if max_samples <= 0:
        raise ValueError("max_samples must be positive.")
    if len(frame) <= max_samples:
        return frame.copy()
    return frame.sample(n=max_samples, random_state=random_state).sort_index()


def _positive_class_values(values: Any, n_features: int) -> np.ndarray:
    """Normalize SHAP output to an ``(n_samples, n_features)`` class-1 array."""
    if isinstance(values, list):
        if len(values) != 2:
            raise ValueError(f"Expected binary SHAP output with two classes, got {len(values)}.")
        values = values[1]

    array = np.asarray(values)
    if array.ndim == 2:
        if array.shape[1] != n_features:
            raise ValueError(f"Unexpected SHAP shape {array.shape}; expected {n_features} features.")
        return array

    if array.ndim == 3:
        # Current SHAP versions commonly return (samples, features, outputs).
        if array.shape[1] == n_features and array.shape[2] >= 2:
            return array[:, :, 1]
        # Defensive support for (samples, outputs, features).
        if array.shape[2] == n_features and array.shape[1] >= 2:
            return array[:, 1, :]

    raise ValueError(
        "Unsupported SHAP value shape for binary classification: "
        f"{array.shape}."
    )


def compute_fold_shap_importance(
    *,
    model: Any,
    model_key: str,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    max_explain_samples: int,
    background_samples: int,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray, pd.DataFrame]:
    """Compute fold-level global SHAP importance using held-out validation rows.

    XGBoost and LightGBM use TreeSHAP. sklearn AdaBoost is not supported by
    TreeExplainer in SHAP 0.50, so it uses the model-agnostic permutation SHAP
    explainer with a deterministic training-background sample.
    """
    explain_frame = _sample_rows(X_validation, max_explain_samples, random_state)
    key = model_key.strip().lower()

    if key in {"xgboost", "lightgbm"}:
        explainer_name = "TreeExplainer"
        explainer = shap.TreeExplainer(model)
        explanation = explainer(explain_frame, check_additivity=False)
        class_values = _positive_class_values(explanation.values, explain_frame.shape[1])
        n_background = 0
    elif key == "adaboost":
        explainer_name = "PermutationExplainer"
        background = _sample_rows(X_train, background_samples, random_state + 10_000)
        masker = shap.maskers.Independent(background, max_samples=len(background))
        explainer = shap.Explainer(
            model.predict_proba,
            masker,
            algorithm="permutation",
            seed=random_state,
        )
        # Permutation SHAP requires at least 2 * num_features + 1 evaluations.
        explanation = explainer(
            explain_frame,
            max_evals=2 * explain_frame.shape[1] + 1,
            silent=True,
        )
        class_values = _positive_class_values(explanation.values, explain_frame.shape[1])
        n_background = len(background)
    else:
        raise ValueError(
            f"Unsupported model key '{model_key}' for SHAP analysis. "
            "Supported models: adaboost, xgboost, lightgbm."
        )

    mean_abs_shap = np.abs(class_values).mean(axis=0)
    importance = pd.DataFrame(
        {
            "feature": explain_frame.columns.astype(str),
            "mean_abs_shap": mean_abs_shap,
        }
    )
    # method='first' gives deterministic unique ranks even when importances tie.
    importance["rank"] = (
        importance["mean_abs_shap"].rank(method="first", ascending=False).astype(int)
    )
    importance = importance.sort_values(["rank", "feature"]).reset_index(drop=True)

    metadata = {
        "explainer": explainer_name,
        "n_explained": len(explain_frame),
        "n_background": n_background,
        "n_features": explain_frame.shape[1],
    }
    return importance, metadata, class_values, explain_frame.reset_index(drop=True)


def _spearman_from_rank_vectors(left: np.ndarray, right: np.ndarray) -> float:
    """Compute Spearman correlation as Pearson correlation of rank vectors."""
    if left.shape != right.shape:
        raise ValueError("Rank vectors must have equal shape.")
    if left.size < 2:
        return float("nan")
    return float(np.corrcoef(left.astype(float), right.astype(float))[0, 1])


def compute_pairwise_rank_stability(
    feature_importance: pd.DataFrame,
    *,
    top_k: int,
) -> pd.DataFrame:
    """Compare every pair of CV folds using Spearman rho and top-k Jaccard overlap."""
    required = {"dataset_key", "dataset_name", "model_key", "model_name", "fold", "feature", "rank"}
    missing = required.difference(feature_importance.columns)
    if missing:
        raise ValueError(f"Missing required SHAP-importance columns: {sorted(missing)}")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    rows: list[dict[str, Any]] = []
    group_cols = ["dataset_key", "dataset_name", "model_key", "model_name"]
    for keys, group in feature_importance.groupby(group_cols, sort=False):
        fold_tables = {
            int(fold): fold_df.set_index("feature")["rank"].sort_index()
            for fold, fold_df in group.groupby("fold")
        }
        for fold_a, fold_b in combinations(sorted(fold_tables), 2):
            a = fold_tables[fold_a]
            b = fold_tables[fold_b]
            shared = a.index.intersection(b.index)
            if len(shared) != len(a) or len(shared) != len(b):
                raise ValueError("Feature sets differ across folds; rank stability is not comparable.")

            spearman = _spearman_from_rank_vectors(
                a.loc[shared].to_numpy(), b.loc[shared].to_numpy()
            )
            top_a = set(a.nsmallest(min(top_k, len(a))).index)
            top_b = set(b.nsmallest(min(top_k, len(b))).index)
            union = top_a | top_b
            jaccard = len(top_a & top_b) / len(union) if union else float("nan")

            rows.append(
                {
                    "dataset_key": keys[0],
                    "dataset_name": keys[1],
                    "model_key": keys[2],
                    "model_name": keys[3],
                    "fold_a": fold_a,
                    "fold_b": fold_b,
                    "n_features": len(shared),
                    "top_k": min(top_k, len(shared)),
                    "spearman_rho": spearman,
                    "top_k_jaccard": jaccard,
                }
            )
    return pd.DataFrame(rows)


def summarize_rank_stability(pairwise_stability: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pairwise fold-stability statistics for RQ3."""
    group_cols = ["dataset_key", "dataset_name", "model_key", "model_name", "top_k"]
    summary = (
        pairwise_stability.groupby(group_cols, as_index=False)
        .agg(
            pair_count=("spearman_rho", "size"),
            spearman_mean=("spearman_rho", "mean"),
            spearman_std=("spearman_rho", "std"),
            spearman_min=("spearman_rho", "min"),
            spearman_max=("spearman_rho", "max"),
            top_k_jaccard_mean=("top_k_jaccard", "mean"),
            top_k_jaccard_std=("top_k_jaccard", "std"),
            top_k_jaccard_min=("top_k_jaccard", "min"),
            top_k_jaccard_max=("top_k_jaccard", "max"),
        )
    )
    return summary


def summarize_consensus_ranking(
    feature_importance: pd.DataFrame,
    *,
    top_k: int,
) -> pd.DataFrame:
    """Build a consensus feature ranking across folds for each dataset/model."""
    frame = feature_importance.copy()
    frame["in_top_k"] = frame["rank"] <= top_k
    group_cols = ["dataset_key", "dataset_name", "model_key", "model_name", "feature"]
    consensus = (
        frame.groupby(group_cols, as_index=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "mean"),
            mean_rank=("rank", "mean"),
            rank_std=("rank", "std"),
            top_k_frequency=("in_top_k", "mean"),
            folds_observed=("fold", "nunique"),
        )
    )
    consensus["consensus_rank"] = (
        consensus.groupby(["dataset_key", "model_key"])["mean_rank"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    return consensus.sort_values(
        ["dataset_key", "model_key", "consensus_rank", "feature"]
    ).reset_index(drop=True)
