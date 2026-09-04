from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau, spearmanr


def compute_fold_shap_importance(model: Any, X: pd.DataFrame) -> pd.Series:
    """Return mean absolute SHAP importance using one method for every model.

    KernelExplainer is used for all classifiers because TreeExplainer does not
    support sklearn's AdaBoostClassifier. The same background and evaluation
    sampling policy is therefore applied to every model in the comparison.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if X.empty:
        raise ValueError("Cannot compute SHAP importance for an empty feature frame.")
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"KernelExplainer requires predict_proba(); {type(model).__name__} does not provide it."
        )

    background = X.iloc[: min(100, len(X))].copy()
    evaluation = X.iloc[: min(200, len(X))].copy()

    def positive_class_probability(values: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(values, columns=X.columns)
        probabilities = np.asarray(model.predict_proba(frame))
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError(
                f"Expected binary predict_proba output with shape (n, 2), got {probabilities.shape}."
            )
        return probabilities[:, 1]

    try:
        explainer = shap.KernelExplainer(positive_class_probability, background)
        values = explainer.shap_values(evaluation)
    except Exception as err:
        raise RuntimeError(
            f"KernelExplainer failed for {type(model).__name__} using the positive-class probability: {err}"
        ) from err

    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    if values.ndim != 2 or values.shape[1] != X.shape[1]:
        raise ValueError(
            f"Expected Kernel SHAP values with shape ({len(evaluation)}, {X.shape[1]}), got {values.shape}."
        )

    return pd.Series(np.abs(values).mean(axis=0), index=X.columns, name="mean_abs_shap")


def summarize_shap_stability(
    fold_importance: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    """Compare feature rankings across folds for each dataset/model pair."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    required = {
        "dataset_key",
        "dataset_name",
        "model_key",
        "model_name",
        "fold",
        "feature",
        "mean_abs_shap",
    }
    missing = required.difference(fold_importance.columns)
    if missing:
        raise ValueError(f"SHAP fold importance is missing required columns: {sorted(missing)}")

    result_columns = [
        "dataset_key",
        "dataset_name",
        "model_key",
        "model_name",
        "n_folds",
        "n_features",
        "top_k",
        "kendall_tau_mean",
        "spearman_rho_mean",
        "top_k_jaccard_mean",
    ]

    if fold_importance.empty:
        return pd.DataFrame(columns=result_columns)

    rows: list[dict[str, Any]] = []
    group_columns = ["dataset_key", "dataset_name", "model_key", "model_name"]
    for group_values, group in fold_importance.groupby(group_columns, sort=False):
        dataset_key, dataset_name, model_key, model_name = group_values
        pivot = group.pivot_table(
            index="fold", columns="feature", values="mean_abs_shap", aggfunc="mean"
        )
        ranks = pivot.rank(axis=1, ascending=False, method="average")
        fold_pairs: list[tuple[float, float, float]] = []
        folds = list(ranks.index)
        effective_top_k = min(top_k, ranks.shape[1])
        for position, first_fold in enumerate(folds):
            for second_fold in folds[position + 1 :]:
                first = ranks.loc[first_fold]
                second = ranks.loc[second_fold]
                valid = first.notna() & second.notna()
                # With fewer than 2 valid features, correlation is undefined;
                # treat as perfect agreement (trivially stable ranking).
                if valid.sum() < 2:
                    kendall = 1.0
                    spearman = 1.0
                else:
                    k_stat = kendalltau(first[valid], second[valid]).statistic
                    s_stat = spearmanr(first[valid], second[valid]).statistic
                    if np.isnan(k_stat):
                        kendall = 1.0 if (first[valid] == second[valid]).all() else 0.0
                    else:
                        kendall = float(k_stat)
                    if np.isnan(s_stat):
                        spearman = 1.0 if (first[valid] == second[valid]).all() else 0.0
                    else:
                        spearman = float(s_stat)

                first_top = set(first.nsmallest(effective_top_k).index)
                second_top = set(second.nsmallest(effective_top_k).index)
                union = first_top | second_top
                jaccard = len(first_top & second_top) / len(union) if union else 1.0
                fold_pairs.append((kendall, spearman, jaccard))

        # With a single fold there are no pairs to compare.
        # Report perfect stability since there is no cross-fold disagreement.
        if fold_pairs:
            pair_values = np.asarray(fold_pairs, dtype=float)
            kendall_mean = float(pair_values[:, 0].mean())
            spearman_mean = float(pair_values[:, 1].mean())
            jaccard_mean = float(pair_values[:, 2].mean())
        else:
            kendall_mean = 1.0
            spearman_mean = 1.0
            jaccard_mean = 1.0

        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset_name": dataset_name,
                "model_key": model_key,
                "model_name": model_name,
                "n_folds": len(folds),
                "n_features": ranks.shape[1],
                "top_k": effective_top_k,
                "kendall_tau_mean": kendall_mean,
                "spearman_rho_mean": spearman_mean,
                "top_k_jaccard_mean": jaccard_mean,
            }
        )

    return pd.DataFrame(rows)