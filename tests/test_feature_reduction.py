"""Feature-reduction tests (scope: 'Test PCA/feature reduction where applicable').

The project's reduction policy is train-fit variance pruning + correlation
pruning applied identically to test (NOT PCA -- see preprocessing.py, the
processed files are already numeric/scaled so a variance/correlation pass is
the deliberate, leakage-safe choice). These tests pin that policy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.preprocessing import apply_feature_selector, fit_feature_selector


def _train_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    base = rng.normal(size=200)
    return pd.DataFrame(
        {
            "target": (rng.normal(size=200) > 0).astype(int),
            "signal": base,
            "duplicate": base + rng.normal(0, 0.01, 200),  # corr ~ 1 with signal
            "noise": rng.normal(size=200),
            "constant_like": np.full(200, 0.5),  # near-zero variance
        }
    )


def test_near_constant_column_is_dropped():
    selector = fit_feature_selector(_train_frame(), "target")

    assert selector.dropped_low_variance == ["constant_like"]


def test_highly_correlated_pair_drops_second_column_only():
    frame = _train_frame()
    selector = fit_feature_selector(frame, "target", correlation_threshold=0.9)

    assert "signal" in selector.kept_columns
    assert "duplicate" in selector.dropped_correlated
    assert selector.dropped_columns.count("signal") == 0


def test_weakly_correlated_columns_are_all_kept():
    frame = pd.DataFrame(
        {
            "target": [0, 1] * 20,
            "a": np.arange(40.0),
            "b": np.arange(-20, 20.0),  # corr(a, b) = -1 -> above abs threshold
            "c": np.array([0, 1, 1, 0] * 10, dtype=float),  # weakly correlated
        }
    )

    selector = fit_feature_selector(frame, "target", correlation_threshold=0.9)

    # a/b are perfectly (anti-)correlated -> exactly one of them is dropped
    assert ("a" in selector.dropped_correlated) ^ ("b" in selector.dropped_correlated)
    assert "c" in selector.kept_columns


def test_target_is_kept_and_excluded_from_checks():
    frame = _train_frame()
    # make the target perfectly correlated with a feature; target must survive
    frame["target"] = (frame["signal"] > 0).astype(int)

    selector = fit_feature_selector(frame, "target")

    assert "target" not in selector.kept_columns  # target is not a *feature*
    assert selector.dropped_columns == [] or "target" not in selector.dropped_columns
    assert "signal" in selector.kept_columns


def test_kept_and_dropped_partition_all_feature_columns():
    frame = _train_frame()

    selector = fit_feature_selector(frame, "target")

    features = [c for c in frame.columns if c != "target"]
    assert sorted(selector.kept_columns + selector.dropped_columns) == sorted(features)


def test_apply_keeps_target_and_kept_columns_in_order():
    frame = _train_frame()
    selector = fit_feature_selector(frame, "target")

    applied = apply_feature_selector(frame, selector, "target")

    assert applied.columns.tolist() == ["target", *selector.kept_columns]


def test_apply_works_on_frame_without_target():
    frame = _train_frame()
    selector = fit_feature_selector(frame, "target")

    applied = apply_feature_selector(frame.drop(columns=["target"]), selector, "target")

    assert applied.columns.tolist() == selector.kept_columns


def test_fit_on_train_applies_identically_to_test():
    """Leakage guard: the selector is fit once on train, then reused on test --
    both splits must end up with exactly the same feature layout."""
    train = _train_frame()
    test = _train_frame().iloc[:50].reset_index(drop=True)

    selector = fit_feature_selector(train, "target")
    train_applied = apply_feature_selector(train, selector, "target")
    test_applied = apply_feature_selector(test, selector, "target")

    assert list(train_applied.columns) == list(test_applied.columns)
