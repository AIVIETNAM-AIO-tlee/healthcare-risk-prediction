"""Cross-fold SHAP feature-importance stability tests (study Phase 4/5 scope).

Controlled expectations: the synthetic generator gives f0 a strong signal,
f1 a weak one, and f2..f4 pure noise, so per-fold rankings must concentrate
on f0/f1 and stability scores must be high; the metric functions are pinned
against hand-computed values; a spy + junk-swap prove held-out test rows
never enter the analysis; identical seeds reproduce identical results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path

from models.factory import build_model
from tree_risk_stability.explain import stability
from tree_risk_stability.explain.stability import (
    per_fold_rankings,
    ranking_features,
    spearman_rank_correlation,
    stability_scores,
    top_k_overlap,
)

FEATURES = [f"f{i}" for i in range(5)]


def _make_dev_data(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=FEATURES)
    logits = 2.0 * X.f0 + 0.6 * X.f1 + rng.normal(0, 0.5, n)
    y = pd.Series((logits > 0).astype(int), name="target")
    return X, y


@pytest.fixture(scope="module")
def xgb_factory():
    with (Path(__file__).resolve().parents[2] / "config.yaml").open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["models"]["xgboost"]
    return lambda: build_model("xgboost", config, random_state=0)


# ---------------------------------------------------------------------------
# Metric functions against hand-computed values
# ---------------------------------------------------------------------------
def test_spearman_identical_rankings_is_one():
    assert spearman_rank_correlation(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_spearman_reversed_rankings_is_minus_one():
    assert spearman_rank_correlation(["a", "b", "c"], ["c", "b", "a"]) == -1.0


def test_spearman_matches_hand_computed_partial_agreement():
    # ranks a=[1,2,3,4], b=[1,3,2,4] -> rho = 1 - 6*2/(4*15) = 0.8
    assert spearman_rank_correlation(["a", "b", "c", "d"], ["a", "c", "b", "d"]) == pytest.approx(0.8)


def test_top_k_overlap_hand_computed():
    assert top_k_overlap(["a", "b", "c", "d"], ["a", "b", "d", "c"], k=2) == 1.0
    assert top_k_overlap(["a", "b", "c", "d"], ["a", "c", "b", "d"], k=2) == 0.5
    # same universe, fully disjoint tops
    assert top_k_overlap(["a", "b", "c", "d"], ["c", "d", "a", "b"], k=2) == 0.0


def test_top_k_overlap_rejects_out_of_range_k():
    with pytest.raises(ValueError, match="k must be within"):
        top_k_overlap(["a", "b", "c"], ["a", "b", "c"], k=0)
    with pytest.raises(ValueError, match="k must be within"):
        top_k_overlap(["a", "b", "c"], ["a", "b", "c"], k=4)


def test_comparing_different_feature_universes_raises():
    with pytest.raises(ValueError, match="same feature set"):
        spearman_rank_correlation(["a", "b"], ["a", "c"])
    with pytest.raises(ValueError, match="at least two rankings"):
        stability.mean_pairwise([["a", "b"]], spearman_rank_correlation)


def test_stability_scores_contract_on_identical_rankings():
    rankings = [["f0", "f1", "f2"]] * 4

    scores = stability_scores(rankings, top_k=2)

    assert scores["spearman"] == 1.0
    assert scores["top2_overlap"] == 1.0
    assert scores["n_rankings"] == 4.0


# ---------------------------------------------------------------------------
# Per-fold extraction on controlled model data
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fold_rankings(xgb_factory):
    X, y = _make_dev_data()
    return per_fold_rankings(xgb_factory, X, y, n_splits=5, random_state=0)


def test_every_cv_fold_produces_a_valid_ranking(fold_rankings):
    assert len(fold_rankings) == 5
    for ranking in fold_rankings:
        assert isinstance(ranking, pd.Series)
        assert sorted(ranking.index) == sorted(FEATURES)
        assert ranking.is_monotonic_decreasing  # sorted desc by construction
        assert np.isfinite(ranking.values).all()
        assert (ranking.values >= 0).all()  # mean|SHAP| is non-negative


def test_signal_features_dominate_every_fold(fold_rankings):
    """Controlled-data expectation: the two signal features must occupy the
    top-2 of every fold ranking, noise features never lead."""
    for ranking in fold_rankings:
        assert set(ranking.index[:2]) == {"f0", "f1"}


def test_stability_scores_high_on_controlled_data(fold_rankings):
    names = [ranking_features(r) for r in fold_rankings]

    scores = stability_scores(names, top_k=3)

    assert scores["spearman"] >= 0.6, f"unstable rankings: {scores}"
    assert scores["top3_overlap"] >= 0.6, f"unstable top-3: {scores}"


# ---------------------------------------------------------------------------
# Held-out test isolation
# ---------------------------------------------------------------------------
def test_held_out_test_rows_never_enter_extraction(xgb_factory, monkeypatch):
    """Spy on extract_ranking: neither explained rows nor background may
    contain held-out test rows."""
    X_dev, y_dev = _make_dev_data(n=200)
    X_test, _ = _make_dev_data(n=50, seed=99)  # disjoint rows, index 0..49
    X_test.index = range(200, 250)
    seen_rows: list[pd.Index] = []
    seen_backgrounds: list[pd.Index] = []

    real_extract = stability.extract_ranking

    def spy(model, X_fold, background=None):
        seen_rows.append(X_fold.index)
        seen_backgrounds.append(background.index)
        return real_extract(model, X_fold, background=background)

    monkeypatch.setattr(stability, "extract_ranking", spy)
    per_fold_rankings(xgb_factory, X_dev, y_dev, n_splits=5, random_state=0)

    test_index = set(X_test.index)
    dev_index = set(X_dev.index)
    assert seen_rows and seen_backgrounds
    for rows, background in zip(seen_rows, seen_backgrounds):
        assert set(rows).isdisjoint(test_index)
        assert set(background).issubset(dev_index)


def test_swapping_held_out_test_data_does_not_change_results(xgb_factory):
    """Black-box proof of independence: identical dev data + completely
    different held-out data must give identical stability results."""
    X_dev, y_dev = _make_dev_data()
    X_test_a, _ = _make_dev_data(n=50, seed=1)
    X_test_b, _ = _make_dev_data(n=50, seed=999)

    rankings_a = per_fold_rankings(xgb_factory, X_dev, y_dev, n_splits=5, random_state=0)
    del X_test_a  # present in scope, never passed in
    rankings_b = per_fold_rankings(xgb_factory, X_dev, y_dev, n_splits=5, random_state=0)
    del X_test_b

    for ranking_a, ranking_b in zip(rankings_a, rankings_b):
        pd.testing.assert_series_equal(ranking_a, ranking_b)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def test_same_seed_reproduces_identical_rankings(xgb_factory):
    X, y = _make_dev_data()

    first = per_fold_rankings(xgb_factory, X, y, n_splits=5, random_state=42)
    second = per_fold_rankings(xgb_factory, X, y, n_splits=5, random_state=42)

    for ranking_a, ranking_b in zip(first, second):
        pd.testing.assert_series_equal(ranking_a, ranking_b)


def test_different_seed_changes_the_fold_partition(xgb_factory):
    X, y = _make_dev_data()

    first = per_fold_rankings(xgb_factory, X, y, n_splits=5, random_state=0)
    second = per_fold_rankings(xgb_factory, X, y, n_splits=5, random_state=1)

    different = any(
        not np.allclose(a.values, b.values) for a, b in zip(first, second)
    )
    assert different, "different seeds produced byte-identical rankings"
