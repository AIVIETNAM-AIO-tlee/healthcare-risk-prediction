"""Reusable model explainability utilities."""

from .shap_analysis import (
    compute_fold_shap_importance,
    compute_pairwise_rank_stability,
    summarize_consensus_ranking,
    summarize_rank_stability,
)

__all__ = [
    "compute_fold_shap_importance",
    "compute_pairwise_rank_stability",
    "summarize_consensus_ranking",
    "summarize_rank_stability",
]
