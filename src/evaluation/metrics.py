from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import auc, f1_score, precision_recall_curve, recall_score, roc_auc_score


METRIC_NAMES = ("roc_auc", "pr_auc", "recall", "f1")


def compute_binary_metrics(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the four metrics used in the research questions.

    ``ROC-AUC`` and ``PR-AUC`` use probabilities and are threshold-independent.
    ``Recall`` and ``F1`` use a fixed decision threshold (0.5 by default), which
    keeps the comparison consistent across models, folds, and datasets.
    """
    y_true_array = np.asarray(y_true, dtype=int)
    y_score_array = np.asarray(y_score, dtype=float)

    if y_true_array.ndim != 1 or y_score_array.ndim != 1:
        raise ValueError("y_true and y_score must both be one-dimensional.")
    if len(y_true_array) != len(y_score_array):
        raise ValueError("y_true and y_score must have the same number of rows.")
    if np.unique(y_true_array).size != 2:
        raise ValueError("Binary metrics require both classes (0 and 1) in y_true.")

    y_pred = (y_score_array >= threshold).astype(int)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true_array, y_score_array)

    return {
        "roc_auc": float(roc_auc_score(y_true_array, y_score_array)),
        "pr_auc": float(auc(recall_curve, precision_curve)),
        "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
    }


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold scores into mean/std/min/max stability summaries."""
    required = {"dataset_key", "dataset_name", "model_key", "model_name", *METRIC_NAMES}
    missing = required.difference(fold_metrics.columns)
    if missing:
        raise ValueError(f"Fold metrics are missing required columns: {sorted(missing)}")

    group_columns = ["dataset_key", "dataset_name", "model_key", "model_name"]
    aggregations: dict[str, list[str]] = {
        metric: ["mean", "std", "min", "max"] for metric in METRIC_NAMES
    }
    summary = fold_metrics.groupby(group_columns, as_index=False).agg(aggregations)
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns.to_flat_index()
    ]

    # Explicit ranges are easy to interpret for RQ2: smaller is more stable.
    for metric in METRIC_NAMES:
        summary[f"{metric}_range"] = summary[f"{metric}_max"] - summary[f"{metric}_min"]

    return summary


def add_dataset_ranks(summary: pd.DataFrame, primary_metric: str = "pr_auc") -> pd.DataFrame:
    """Rank models within each dataset by mean CV performance.

    The primary metric is descending. ROC-AUC and F1 are deterministic
    tie-breakers, while the primary metric's standard deviation is used as a
    final stability tie-breaker (lower is better).
    """
    if primary_metric not in METRIC_NAMES:
        raise ValueError(f"Unsupported primary metric: {primary_metric}")

    primary_mean = f"{primary_metric}_mean"
    primary_std = f"{primary_metric}_std"
    required = {"dataset_key", primary_mean, primary_std, "roc_auc_mean", "f1_mean"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Summary is missing required columns: {sorted(missing)}")

    ranked_parts: list[pd.DataFrame] = []
    for _, dataset_frame in summary.groupby("dataset_key", sort=False):
        ranked = dataset_frame.sort_values(
            by=[primary_mean, "roc_auc_mean", "f1_mean", primary_std],
            ascending=[False, False, False, True],
        ).copy()
        ranked.insert(4, "cv_rank", np.arange(1, len(ranked) + 1))
        ranked_parts.append(ranked)

    return pd.concat(ranked_parts, ignore_index=True)


def build_overall_comparison(ranked_summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize each model across all datasets without pooling patient rows.

    Since the datasets represent different prediction problems and class
    prevalences, we report an average within-dataset rank plus the unweighted
    mean of each dataset-level metric. This prevents the largest dataset from
    dominating the cross-dataset comparison.
    """
    aggregations = {
        "cv_rank": "mean",
        **{f"{metric}_mean": "mean" for metric in METRIC_NAMES},
        **{f"{metric}_std": "mean" for metric in METRIC_NAMES},
    }
    overall = (
        ranked_summary.groupby(["model_key", "model_name"], as_index=False)
        .agg(aggregations)
        .rename(columns={"cv_rank": "mean_dataset_rank"})
    )
    return overall.sort_values(
        by=["mean_dataset_rank", "pr_auc_mean", "roc_auc_mean"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
