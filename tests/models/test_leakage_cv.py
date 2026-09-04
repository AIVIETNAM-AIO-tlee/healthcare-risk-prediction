"""Leakage Vectors 3 & 4 — CV runner isolation and inner-fold validity.

Vector 3: inner grid search must only ever see outer-train rows. A row-recording
spy estimator proves no test-fold row reaches any inner fit.
Vector 4: build_inner_cv(config) must guarantee >=2 minority samples per INNER
validation fold for the synthetic regime (phase-03 inner-CV override).

Plan refs: phase-03-model-pipeline.md Requirements (inner grid search on train
folds only; per-dataset inner-CV override so each inner validation fold holds
>=2 minority samples), phase-03 step 5(b) leakage guard.

RED contract: run_nested_cv / make_model implemented in Phase 3;
build_inner_cv + make_spy_fit_context are QA-mandated seams that make the
isolation properties testable in isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qa_helpers import make_multiclass_mini


class RowSpyEstimator:
    """Minimal classifier that records which row-indices it was fit on."""

    def __init__(self):
        self.seen_indices_: list = []
        self.classes_ = np.array([0, 1])

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def fit(self, X, y=None, sample_weight=None):
        idx = X.index if hasattr(X, "index") else range(len(X))
        self.seen_indices_.extend(list(idx))
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.full((len(X), 2), 0.5)

    def score(self, X, y):
        return 0.5


def _outer_split_indices(n: int = 60, seed: int = 0):
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(seed)
    y = pd.Series((rng.normal(0, 1, n) > 0).astype(int))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(iter(skf.split(np.zeros(n), y)))
    return train_idx, test_idx, y


class TestInnerSearchSeesTrainOnly:
    """Vector 3: rows seen during inner tuning ∩ outer-test indices == empty."""

    def test_inner_fits_never_touch_test_rows(self):
        from tree_risk_stability.models.nested_cv import make_spy_fit_context

        train_idx, test_idx, y = _outer_split_indices()
        ctx = make_spy_fit_context(train_idx=train_idx, estimator_class=RowSpyEstimator)
        ctx.run_inner_search(pd.DataFrame({"f": range(60)}), y)
        seen = set(ctx.spy.seen_indices_)
        assert seen.isdisjoint(set(test_idx)), (
            f"inner search touched {len(seen & set(test_idx))} outer-test rows"
        )

    def test_spy_actually_fits_something(self):
        """Guard against vacuous pass: the context must have performed >=2 inner fits."""
        from tree_risk_stability.models.nested_cv import make_spy_fit_context

        train_idx, _, y = _outer_split_indices()
        ctx = make_spy_fit_context(train_idx=train_idx, estimator_class=RowSpyEstimator)
        ctx.run_inner_search(pd.DataFrame({"f": range(60)}), y)
        assert len(ctx.spy.seen_indices_) >= 2, "spy saw no inner fits — test is vacuous"


class TestInnerFoldMinorityStrata:
    """Vector 4: synthetic regime must never produce singleton minority inner folds."""

    def test_build_inner_cv_exists_and_yields_valid_splits(self):
        from tree_risk_stability.models.nested_cv import build_inner_cv

        X, y = make_multiclass_mini()  # Low=6, Medium=3, High=1 → override REQUIRED
        cv = build_inner_cv(y=y, config={"slug": "synthetic", "inner_cv_override": True})
        minority_counts = [
            int((y.iloc[val_idx] == "High").sum()) for train_idx, val_idx in cv.split(np.zeros(len(y)), y)
        ]
        assert min(minority_counts) >= 2, (
            f"inner validation folds contain singleton minority strata: {minority_counts}"
        )

    def test_default_binary_config_uses_plain_stratification(self):
        """Binary slugs keep standard StratifiedKFold(3) — override is per-dataset only."""
        from tree_risk_stability.models.nested_cv import build_inner_cv

        _, _, y = _outer_split_indices()
        cv = build_inner_cv(y=y, config={"slug": "heart", "inner_cv_override": False})
        assert cv.get_n_splits() == 3
