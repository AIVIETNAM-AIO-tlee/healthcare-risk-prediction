from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


_REQUIRED_COLUMNS = {
    "dataset_key",
    "dataset_name",
    "model_key",
    "model_name",
    "fold",
    "feature",
    "mean_abs_shap",
    "rank",
}


def _validate_feature_importance(feature_importance: pd.DataFrame) -> None:
    missing = _REQUIRED_COLUMNS.difference(feature_importance.columns)
    if missing:
        raise ValueError(
            "SHAP feature importance is missing required columns: "
            f"{sorted(missing)}"
        )


def _safe_statistic(statistic: float, left: pd.Series, right: pd.Series) -> float:
    if not np.isnan(statistic):
        return float(statistic)
    return 1.0 if left.equals(right) else 0.0


def compute_pairwise_rank_stability(
    feature_importance: pd.DataFrame,
    *,
    top_k: int,
) -> pd.DataFrame:
    """Compare every pair of CV folds for each dataset/model pair.

    The comparison reports:
    - Kendall's tau over the complete feature ranking.
    - Spearman's rho over the complete feature ranking.
    - Jaccard overlap of the top-k feature sets.
    """
    _validate_feature_importance(feature_importance)
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if feature_importance.empty:
        return pd.DataFrame(
            columns=[
                "dataset_key",
                "dataset_name",
                "model_key",
                "model_name",
                "fold_a",
                "fold_b",
                "n_features",
                "top_k",
                "kendall_tau",
                "spearman_rho",
                "top_k_jaccard",
            ]
        )

    rows: list[dict[str, Any]] = []
    group_columns = ["dataset_key", "dataset_name", "model_key", "model_name"]
    for group_values, group in feature_importance.groupby(group_columns, sort=False):
        fold_tables = {
            int(fold): (
                fold_df[["feature", "rank", "mean_abs_shap"]]
                .drop_duplicates(subset=["feature"])
                .set_index("feature")
                .sort_index()
            )
            for fold, fold_df in group.groupby("fold", sort=True)
        }
        folds = sorted(fold_tables)
        for fold_a, fold_b in combinations(folds, 2):
            first = fold_tables[fold_a]
            second = fold_tables[fold_b]
            shared = first.index.intersection(second.index)
            if len(shared) != len(first.index) or len(shared) != len(second.index):
                raise ValueError(
                    "Feature sets differ across folds; SHAP rank stability is not comparable."
                )

            first_rank = first.loc[shared, "rank"]
            second_rank = second.loc[shared, "rank"]
            if len(shared) < 2:
                kendall = 1.0
                spearman = 1.0
            else:
                kendall = _safe_statistic(
                    kendalltau(first_rank, second_rank).statistic,
                    first_rank,
                    second_rank,
                )
                spearman = _safe_statistic(
                    spearmanr(first_rank, second_rank).statistic,
                    first_rank,
                    second_rank,
                )

            effective_top_k = min(top_k, len(shared))
            first_top = set(first_rank.nsmallest(effective_top_k).index)
            second_top = set(second_rank.nsmallest(effective_top_k).index)
            union = first_top | second_top
            jaccard = len(first_top & second_top) / len(union) if union else 1.0

            rows.append(
                {
                    "dataset_key": group_values[0],
                    "dataset_name": group_values[1],
                    "model_key": group_values[2],
                    "model_name": group_values[3],
                    "fold_a": fold_a,
                    "fold_b": fold_b,
                    "n_features": len(shared),
                    "top_k": effective_top_k,
                    "kendall_tau": float(kendall),
                    "spearman_rho": float(spearman),
                    "top_k_jaccard": float(jaccard),
                }
            )

    return pd.DataFrame(rows)


def summarize_rank_stability(pairwise_stability: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold-pair stability metrics for each dataset/model pair."""
    required = {
        "dataset_key",
        "dataset_name",
        "model_key",
        "model_name",
        "top_k",
        "kendall_tau",
        "spearman_rho",
        "top_k_jaccard",
    }
    missing = required.difference(pairwise_stability.columns)
    if missing:
        raise ValueError(
            "Pairwise SHAP stability is missing required columns: "
            f"{sorted(missing)}"
        )

    if pairwise_stability.empty:
        return pd.DataFrame(
            columns=[
                "dataset_key",
                "dataset_name",
                "model_key",
                "model_name",
                "pair_count",
                "n_folds",
                "n_features",
                "top_k",
                "kendall_tau_mean",
                "kendall_tau_std",
                "kendall_tau_min",
                "kendall_tau_max",
                "spearman_mean",
                "spearman_std",
                "spearman_min",
                "spearman_max",
                "top_k_jaccard_mean",
                "top_k_jaccard_std",
                "top_k_jaccard_min",
                "top_k_jaccard_max",
            ]
        )

    group_cols = ["dataset_key", "dataset_name", "model_key", "model_name", "top_k"]
    summary = (
        pairwise_stability.groupby(group_cols, as_index=False)
        .agg(
            pair_count=("spearman_rho", "size"),
            n_features=("n_features", "first"),
            kendall_tau_mean=("kendall_tau", "mean"),
            kendall_tau_std=("kendall_tau", "std"),
            kendall_tau_min=("kendall_tau", "min"),
            kendall_tau_max=("kendall_tau", "max"),
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

    # Solve n_folds from the number of pairwise combinations: nC2 = pair_count.
    summary["n_folds"] = (
        (1 + np.sqrt(1 + 8 * summary["pair_count"].astype(float))) / 2
    ).round().astype(int)

    for column in [
        "kendall_tau_std",
        "spearman_std",
        "top_k_jaccard_std",
    ]:
        summary[column] = summary[column].fillna(0.0)

    ordered = [
        "dataset_key",
        "dataset_name",
        "model_key",
        "model_name",
        "pair_count",
        "n_folds",
        "n_features",
        "top_k",
        "kendall_tau_mean",
        "kendall_tau_std",
        "kendall_tau_min",
        "kendall_tau_max",
        "spearman_mean",
        "spearman_std",
        "spearman_min",
        "spearman_max",
        "top_k_jaccard_mean",
        "top_k_jaccard_std",
        "top_k_jaccard_min",
        "top_k_jaccard_max",
    ]
    return summary[ordered]


def build_rank_matrix(
    feature_importance: pd.DataFrame,
    *,
    dataset_key: str,
    model_key: str,
    max_features: int | None = None,
) -> pd.DataFrame:
    """Return a fold x feature matrix of feature ranks for heatmap visualization."""
    _validate_feature_importance(feature_importance)
    group = feature_importance[
        (feature_importance["dataset_key"] == dataset_key)
        & (feature_importance["model_key"] == model_key)
    ].copy()
    if group.empty:
        raise ValueError(
            f"No SHAP feature-importance rows found for dataset='{dataset_key}', model='{model_key}'."
        )

    summary = (
        group.groupby("feature", as_index=False)
        .agg(mean_rank=("rank", "mean"), mean_abs_shap=("mean_abs_shap", "mean"))
        .sort_values(["mean_rank", "mean_abs_shap", "feature"], ascending=[True, False, True])
    )
    if max_features is not None:
        summary = summary.head(max_features)
    ordered_features = summary["feature"].tolist()

    matrix = group.pivot_table(index="fold", columns="feature", values="rank", aggfunc="mean")
    matrix = matrix.reindex(columns=ordered_features)
    matrix = matrix.sort_index(axis=0).sort_index(axis=1, key=lambda idx: [ordered_features.index(c) for c in idx])
    matrix.index.name = "fold"
    return matrix


def build_fold_correlation_matrix(
    feature_importance: pd.DataFrame,
    *,
    dataset_key: str,
    model_key: str,
    method: str = "spearman",
) -> pd.DataFrame:
    """Return a fold x fold correlation matrix for rank vectors."""
    if method not in {"spearman", "kendall"}:
        raise ValueError("method must be either 'spearman' or 'kendall'.")
    _validate_feature_importance(feature_importance)

    group = feature_importance[
        (feature_importance["dataset_key"] == dataset_key)
        & (feature_importance["model_key"] == model_key)
    ].copy()
    if group.empty:
        raise ValueError(
            f"No SHAP feature-importance rows found for dataset='{dataset_key}', model='{model_key}'."
        )

    ranks = group.pivot_table(index="feature", columns="fold", values="rank", aggfunc="mean")
    ranks = ranks.sort_index(axis=0).sort_index(axis=1)
    folds = list(ranks.columns)
    matrix = pd.DataFrame(np.eye(len(folds), dtype=float), index=folds, columns=folds)

    for fold_a, fold_b in combinations(folds, 2):
        left = ranks[fold_a]
        right = ranks[fold_b]
        valid = left.notna() & right.notna()
        if valid.sum() < 2:
            stat = 1.0
        elif method == "spearman":
            stat = _safe_statistic(spearmanr(left[valid], right[valid]).statistic, left[valid], right[valid])
        else:
            stat = _safe_statistic(kendalltau(left[valid], right[valid]).statistic, left[valid], right[valid])
        matrix.loc[fold_a, fold_b] = float(stat)
        matrix.loc[fold_b, fold_a] = float(stat)

    matrix.index = [f"Fold {int(i)}" for i in matrix.index]
    matrix.columns = [f"Fold {int(i)}" for i in matrix.columns]
    return matrix
