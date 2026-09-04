"""Class-imbalance handling tests (scope: 'Test class imbalance handling').

The project's one imbalance policy (config.yaml ``balance_training``) is
sklearn's "balanced" sample weighting applied at fit time. The preprocessing
helper exposes the same numbers for EDA/reporting, so the tests pin the
policy's arithmetic and its equivalence with the weights actually used in
training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from data.preprocessing import compute_class_weights


def _imbalanced_target() -> pd.Series:
    # 10 rows, 8 negative / 2 positive -> w0 = 10/(2*8), w1 = 10/(2*2)
    return pd.Series([0, 1, 0, 0, 0, 0, 0, 0, 1, 0], name="target")


def test_weights_follow_sklearn_balanced_formula():
    weights = compute_class_weights(_imbalanced_target())

    assert weights[0] == 10 / (2 * 8)
    assert weights[1] == 10 / (2 * 2)


def test_minority_class_receives_larger_weight():
    weights = compute_class_weights(_imbalanced_target())

    assert weights[1] > weights[0]
    # weight ratio is the inverse of the count ratio (8/2)
    assert weights[1] / weights[0] == 8 / 2


def test_balanced_classes_receive_equal_weight():
    y = pd.Series([0, 1, 0, 1, 0, 1])

    weights = compute_class_weights(y)

    assert weights[0] == weights[1] == 1.0


def test_returns_plain_dict_of_floats_keyed_by_class():
    weights = compute_class_weights(_imbalanced_target())

    assert set(weights.keys()) == {0, 1}
    assert all(isinstance(value, float) for value in weights.values())


def test_matches_sample_weights_actually_applied_at_fit_time():
    """Policy equivalence: per-row 'balanced' sample weights used by
    ``run_models._fit_model`` must equal the reported per-class weights."""
    y = _imbalanced_target()

    weights = compute_class_weights(y)
    per_row = compute_sample_weight(class_weight="balanced", y=y)

    expected = np.array([weights[value] for value in y])
    np.testing.assert_allclose(per_row, expected)
