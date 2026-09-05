"""Cross-fold SHAP feature-importance stability analysis (study Phase 4/5).

Rankings are extracted per CV fold from models fit on that fold's train rows
only; stability is summarized as mean pairwise Spearman rank correlation and
mean pairwise top-K overlap. Held-out test data is structurally out of reach:
`per_fold_rankings` accepts development data only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

from tree_risk_stability.explain.shap_extract import extract_ranking


def ranking_features(ranking: pd.Series) -> list[str]:
    """Ranking Series (feature -> importance, sorted desc) -> ordered names."""
    return [str(feature) for feature in ranking.index]


def _validate_same_universe(rankings: list[list[str]]) -> None:
    if len(rankings) < 2:
        raise ValueError("Ranking comparison needs at least two rankings.")
    universe = set(rankings[0])
    for ranking in rankings[1:]:
        if set(ranking) != universe:
            raise ValueError("Rankings must cover the same feature set.")


def spearman_rank_correlation(ranking_a: list[str], ranking_b: list[str]) -> float:
    """Spearman correlation between two rankings of the same features.

    Positions are 1-based (best feature = 1), so identical rankings correlate
    1.0 and exactly reversed rankings correlate -1.0.
    """
    _validate_same_universe([ranking_a, ranking_b])
    positions_a = np.array([ranking_a.index(feature) for feature in ranking_a], dtype=float)
    positions_b = np.array([ranking_a.index(feature) for feature in ranking_b], dtype=float)
    correlation, _ = spearmanr(positions_a, positions_b)
    return float(correlation)


def top_k_overlap(ranking_a: list[str], ranking_b: list[str], k: int) -> float:
    """|top-K(a) ∩ top-K(b)| / k for rankings of the same features."""
    _validate_same_universe([ranking_a, ranking_b])
    if not 1 <= k <= len(ranking_a):
        raise ValueError(f"k must be within [1, {len(ranking_a)}], got {k}.")
    top_a = set(ranking_a[:k])
    top_b = set(ranking_b[:k])
    return len(top_a & top_b) / k


def mean_pairwise(rankings: list[list[str]], metric) -> float:
    """Mean of `metric` over all unordered ranking pairs."""
    _validate_same_universe(rankings)
    scores = [
        metric(rankings[i], rankings[j])
        for i in range(len(rankings))
        for j in range(i + 1, len(rankings))
    ]
    return float(np.mean(scores))


def stability_scores(
    rankings: list[list[str]], top_k: int = 3
) -> dict[str, float]:
    """Summarize cross-fold ranking stability.

    Returns mean pairwise Spearman rank correlation and mean pairwise top-K
    overlap. Both are 1.0 when every fold produces an identical ranking.
    """
    feature_count = len(rankings[0])
    default_k = min(top_k, feature_count) if top_k else feature_count
    return {
        "spearman": mean_pairwise(rankings, spearman_rank_correlation),
        f"top{default_k}_overlap": mean_pairwise(
            rankings, lambda a, b: top_k_overlap(a, b, default_k)
        ),
        "n_rankings": float(len(rankings)),
    }


def per_fold_rankings(
    model_factory,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    n_splits: int = 5,
    random_state: int = 0,
) -> list[pd.Series]:
    """Fit + explain per CV fold, touching development rows only.

    Each fold fits a fresh model on its train rows and extracts the ranking on
    its validation rows with the fold's train rows as interventional
    background. The held-out test set is never an argument here, so it cannot
    leak into the stability analysis.
    """
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rankings: list[pd.Series] = []
    for train_indices, validation_indices in splitter.split(X_dev, y_dev):
        X_fold_train = X_dev.iloc[train_indices]
        y_fold_train = y_dev.iloc[train_indices]
        model = model_factory()
        model.fit(X_fold_train, y_fold_train)
        rankings.append(
            extract_ranking(model, X_dev.iloc[validation_indices], background=X_fold_train)
        )
    return rankings
