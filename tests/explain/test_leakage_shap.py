"""Leakage Vector 6 — SHAP background must originate from the TRAIN fold.

Interventional TreeExplainer integrates against a background distribution. If that
background included test-fold rows, explanations absorb test information — a
subtle explainability leakage channel.

Plan refs: phase-04-explainability.md Risk Assessment (interventional pinned,
train-fold background); phase-03 Requirements (fitted estimators not persisted —
extraction happens in-process right after fit, so background = train fold by
construction IF the seam enforces it).

RED contract: extract_ranking implemented in Phase 4. The signature
`extract_ranking(model, X_fold, background=None)` makes the provenance of
`background` an explicit, checkable argument instead of an internal choice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _tiny_fitted_tree(X: pd.DataFrame, y: pd.Series):
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    return model


class TestBackgroundProvenance:
    def test_signature_exposes_background_parameter(self):
        """The QA-mandated seam: background is an explicit argument so callers can
        be audited for train-fold-only provenance."""
        import inspect

        from tree_risk_stability.explain.shap_extract import extract_ranking

        params = inspect.signature(extract_ranking).parameters
        assert "background" in params, (
            "extract_ranking must take `background` explicitly; implicit internal "
            "background selection cannot be audited for train-fold-only provenance"
        )

    def test_background_changes_attributions(self):
        """Different backgrounds → different SHAP values under interventional
        perturbation. Proves the background actually participates, i.e., feeding it
        leaked rows WOULD change results — hence it must be constrained to train."""
        rng = np.random.default_rng(9)
        n = 40
        X_train = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
        y_train = pd.Series((X_train.a + X_train.b + rng.normal(0, 0.3, n) > 0).astype(int))
        model = _tiny_fitted_tree(X_train, y_train)

        from tree_risk_stability.explain.shap_extract import extract_ranking

        bg_a = X_train.iloc[:20]
        bg_b = pd.DataFrame(
            {"a": rng.normal(5, 1, 20), "b": rng.normal(-5, 1, 20)}
        )
        probe = X_train.iloc[:5]
        rank_a = extract_ranking(model, probe, background=bg_a)
        rank_b = extract_ranking(model, probe, background=bg_b)
        assert not np.allclose(rank_a.values, rank_b.values), (
            "background had no effect on attributions — interventional path not engaged "
            "or background ignored; provenance constraint would be unenforceable"
        )
