"""Unit tests for the ACM3-23 preprocessing additions in src/data/preprocessing.py:
class-weight reporting, IQR outlier clipping, and feature reduction.

Covers: class-weight correctness against sklearn, IQR bound fitting/train-only
leakage-safety, NaN-safety of clipping, low-variance + correlated-pair feature
selection, target exclusion, and input immutability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.preprocessing import (
	FittedFeatureSelector,
	FittedOutlierBounds,
	apply_feature_selector,
	apply_outlier_bounds,
	compute_class_weights,
	fit_feature_selector,
	fit_outlier_bounds,
)

TARGET = "target"


# ---------------------------------------------------------------------------
# compute_class_weights
# ---------------------------------------------------------------------------

def test_compute_class_weights_matches_sklearn_balanced_formula():
	y = pd.Series([0, 0, 0, 0, 1, 1])  # 4 negatives, 2 positives
	weights = compute_class_weights(y)
	# sklearn "balanced": n_samples / (n_classes * n_samples_per_class)
	assert weights[0] == pytest.approx(6 / (2 * 4))
	assert weights[1] == pytest.approx(6 / (2 * 2))


def test_compute_class_weights_balanced_target_gives_equal_weights():
	y = pd.Series([0, 1, 0, 1])
	weights = compute_class_weights(y)
	assert weights[0] == pytest.approx(1.0)
	assert weights[1] == pytest.approx(1.0)


def test_compute_class_weights_keys_are_python_ints():
	y = pd.Series([0, 1, 1])
	weights = compute_class_weights(y)
	assert set(weights.keys()) == {0, 1}
	assert all(isinstance(key, int) for key in weights)


# ---------------------------------------------------------------------------
# fit_outlier_bounds / apply_outlier_bounds
# ---------------------------------------------------------------------------

def test_fit_outlier_bounds_matches_manual_tukey_fence():
	df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]})
	fitted = fit_outlier_bounds(df, ["x"], multiplier=1.5)
	q1, q3 = df["x"].quantile(0.25), df["x"].quantile(0.75)
	iqr = q3 - q1
	expected_lower, expected_upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
	lower, upper = fitted.bounds["x"]
	assert lower == pytest.approx(expected_lower)
	assert upper == pytest.approx(expected_upper)


def test_apply_outlier_bounds_clips_values_outside_range():
	fitted = FittedOutlierBounds(bounds={"x": (0.0, 10.0)})
	df = pd.DataFrame({"x": [-5.0, 0.0, 5.0, 10.0, 999.0]})
	out = apply_outlier_bounds(df, fitted)
	assert out["x"].tolist() == [0.0, 0.0, 5.0, 10.0, 10.0]


def test_apply_outlier_bounds_preserves_nan():
	fitted = FittedOutlierBounds(bounds={"x": (0.0, 10.0)})
	df = pd.DataFrame({"x": [np.nan, 999.0]})
	out = apply_outlier_bounds(df, fitted)
	assert np.isnan(out["x"].iloc[0])
	assert out["x"].iloc[1] == 10.0


def test_apply_outlier_bounds_does_not_mutate_input():
	fitted = FittedOutlierBounds(bounds={"x": (0.0, 10.0)})
	df = pd.DataFrame({"x": [999.0]})
	before = df.copy(deep=True)
	apply_outlier_bounds(df, fitted)
	pd.testing.assert_frame_equal(df, before)


def test_outlier_bounds_fit_on_train_not_leaked_from_test():
	"""Bounds fit on train must ignore test-split extreme values."""
	train = pd.DataFrame({"x": [10.0, 11.0, 12.0, 13.0, 14.0]})
	test = pd.DataFrame({"x": [-1000.0, 1000.0]})
	fitted = fit_outlier_bounds(train, ["x"], multiplier=1.5)
	clipped_test = apply_outlier_bounds(test, fitted)
	lower, upper = fitted.bounds["x"]
	# Bounds are derived purely from train's tight range, so test's extreme
	# values get clipped hard -- they never influenced the bounds themselves.
	assert clipped_test["x"].iloc[0] == pytest.approx(lower)
	assert clipped_test["x"].iloc[1] == pytest.approx(upper)
	assert lower > 0  # train values are all in [10, 14]; test's -1000 played no part


# ---------------------------------------------------------------------------
# fit_feature_selector / apply_feature_selector
# ---------------------------------------------------------------------------

def test_low_variance_column_is_dropped():
	df = pd.DataFrame(
		{
			TARGET: [0, 1, 0, 1],
			"varies": [1.0, 2.0, 3.0, 4.0],
			"constant": [5.0, 5.0, 5.0, 5.0],
		}
	)
	fitted = fit_feature_selector(df, TARGET, variance_threshold=1e-4, correlation_threshold=0.9)
	assert "constant" in fitted.dropped_low_variance
	assert "varies" in fitted.kept_columns


def test_highly_correlated_pair_drops_second_column_only():
	base = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
	df = pd.DataFrame(
		{
			TARGET: [0, 1, 0, 1, 0, 1, 0, 1],
			"a": base,
			"b": [v * 2 for v in base],  # perfectly correlated with 'a'
			"c": [8, 1, 7, 2, 6, 3, 5, 4],  # not correlated with a/b
		}
	)
	fitted = fit_feature_selector(df, TARGET, variance_threshold=1e-4, correlation_threshold=0.9)
	assert "a" in fitted.kept_columns
	assert "b" in fitted.dropped_correlated
	assert "c" in fitted.kept_columns


def test_target_column_never_dropped_or_checked():
	# Target itself is constant here (would trip the variance check if not excluded).
	df = pd.DataFrame({TARGET: [1, 1, 1, 1], "x": [1.0, 2.0, 3.0, 4.0]})
	fitted = fit_feature_selector(df, TARGET)
	assert TARGET not in fitted.dropped_columns
	assert TARGET not in fitted.kept_columns  # target isn't a "feature" at all


def test_apply_feature_selector_keeps_target_first_and_drops_selected_columns():
	df = pd.DataFrame({TARGET: [0, 1], "keep": [1.0, 2.0], "drop": [5.0, 5.0]})
	fitted = FittedFeatureSelector(kept_columns=["keep"], dropped_low_variance=["drop"], dropped_correlated=[])
	out = apply_feature_selector(df, fitted, TARGET)
	assert list(out.columns) == [TARGET, "keep"]


def test_apply_feature_selector_applies_train_fitted_columns_to_test():
	"""A column dropped based on train statistics must also disappear from test,
	even if test's own data would not have triggered the same drop."""
	train = pd.DataFrame(
		{TARGET: [0, 1, 0, 1], "a": [1.0, 2.0, 3.0, 4.0], "const_in_train": [5.0, 5.0, 5.0, 5.0]}
	)
	test = pd.DataFrame(
		{TARGET: [0, 1], "a": [1.5, 2.5], "const_in_train": [1.0, 9.0]}  # varies in test!
	)
	fitted = fit_feature_selector(train, TARGET)
	out = apply_feature_selector(test, fitted, TARGET)
	assert "const_in_train" not in out.columns
	assert list(out.columns) == [TARGET, "a"]


def test_no_columns_dropped_when_nothing_qualifies():
	df = pd.DataFrame({TARGET: [0, 1, 0, 1], "a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 1.0, 3.0, 2.0]})
	fitted = fit_feature_selector(df, TARGET, variance_threshold=1e-4, correlation_threshold=0.9)
	assert fitted.dropped_columns == []
	assert set(fitted.kept_columns) == {"a", "b"}
