"""Evaluation metrics and model-comparison helpers."""

from .metrics import (
    METRIC_NAMES,
    add_dataset_ranks,
    build_overall_comparison,
    compute_binary_metrics,
    summarize_fold_metrics,
)

__all__ = [
    "METRIC_NAMES",
    "add_dataset_ranks",
    "build_overall_comparison",
    "compute_binary_metrics",
    "summarize_fold_metrics",
]
