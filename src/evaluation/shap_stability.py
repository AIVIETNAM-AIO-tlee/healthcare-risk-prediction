from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def _leaf_pos_prob(val_counts: np.ndarray, node_idx: int, classes: np.ndarray) -> float:
    """Calculate positive class probability for a decision tree node."""
    counts = val_counts[node_idx][0]
    total = counts.sum()
    if total == 0:
        return 0.0
    if len(counts) > 1:
        pos_idx = np.where(classes == 1)[0]
        idx = int(pos_idx[0]) if len(pos_idx) > 0 else (len(counts) - 1)
        return float(counts[idx] / total)
    return 1.0 if (len(classes) > 0 and classes[0] == 1) else 0.0


def _compute_adaboost_tree_shap(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Compute exact Tree SHAP values for an AdaBoostClassifier of decision trees.

    AdaBoost produces an additive model F(x) = sum_m w_m * h_m(x).
    By SHAP additivity: phi_i(F) = sum_m w_m * phi_i(h_m).
    For decision stumps (depth <= 1), each tree only splits on a single feature,
    allowing vectorized computation with zero explainer overhead.
    For deeper trees, TreeExplainer is applied to each constituent estimator.
    """
    import shap
    from sklearn.tree import DecisionTreeClassifier

    estimators = getattr(model, "estimators_", None)
    weights = getattr(model, "estimator_weights_", None)
    if estimators is None or weights is None or len(estimators) == 0:
        raise ValueError("AdaBoost model has no fitted estimators or weights.")

    n_samples, n_features = X.shape
    total_shap = np.zeros((n_samples, n_features), dtype=float)

    is_all_stumps = all(
        isinstance(e, DecisionTreeClassifier) and getattr(e.tree_, "max_depth", 0) <= 1
        for e in estimators
    )

    if is_all_stumps:
        X_mat = X.to_numpy()
        for tree, weight in zip(estimators, weights):
            feat = tree.tree_.feature[0]
            if feat < 0:
                continue
            thresh = tree.tree_.threshold[0]
            val_counts = tree.tree_.value
            classes = tree.classes_
            val_root = _leaf_pos_prob(val_counts, 0, classes)
            val_left = _leaf_pos_prob(val_counts, 1, classes)
            val_right = _leaf_pos_prob(val_counts, 2, classes)

            mask = X_mat[:, feat] <= thresh
            total_shap[mask, feat] += weight * (val_left - val_root)
            total_shap[~mask, feat] += weight * (val_right - val_root)
        return total_shap
    else:
        for tree, weight in zip(estimators, weights):
            tree_vals = shap.TreeExplainer(tree).shap_values(X)
            if isinstance(tree_vals, list):
                tree_vals = tree_vals[-1]
            elif tree_vals.ndim == 3:
                tree_vals = tree_vals[:, :, -1]
            total_shap += float(weight) * tree_vals
        return total_shap


def compute_fold_shap_importance(model: Any, X: pd.DataFrame) -> pd.Series:
    """Return mean absolute SHAP importance for one fitted binary classifier."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is required for explanation stability. Install requirements.txt first."
        ) from exc

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # 1. Check if model is an AdaBoost tree ensemble
    if hasattr(model, "estimators_") and hasattr(model, "estimator_weights_"):
        try:
            values = _compute_adaboost_tree_shap(model, X)
            return pd.Series(np.abs(values).mean(axis=0), index=X.columns, name="mean_abs_shap")
        except Exception:
            pass  # Fallback to standard flow if unexpected structure

    # 2. Try native TreeExplainer (XGBoost, LightGBM, RandomForest, etc.)
    try:
        values = shap.TreeExplainer(model).shap_values(X)
    except Exception:
        # 3. Model-agnostic KernelExplainer fallback
        try:
            background = shap.sample(X, min(100, len(X)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            eval_X = shap.sample(X, min(200, len(X))) if len(X) > 200 else X
            values = explainer.shap_values(eval_X)
        except Exception as generic_error:
            raise RuntimeError(
                f"Unable to compute SHAP values for {type(model).__name__}."
            ) from generic_error

    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    if values.ndim != 2 or values.shape[1] != X.shape[1]:
        raise ValueError(
            f"Expected SHAP values with shape (n_samples, {X.shape[1]}), got {values.shape}."
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