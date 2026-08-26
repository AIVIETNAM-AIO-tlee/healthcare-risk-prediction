"""Unit tests for src/data/preprocessing.py.

Covers: missing-value handling, encoding behavior (binary + one-hot),
scaling, feature/target separation, input-frame immutability, output
dimensions, determinism across repeated runs, and invalid inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import CATEGORICAL, NUMERIC

from data.preprocessing import (
	FittedPreprocessor,
	clean_target,
	fit_preprocessor,
	mark_invalid_zeros_as_missing,
	transform_features,
)

TARGET = "target"


# ---------------------------------------------------------------------------
# mark_invalid_zeros_as_missing
# ---------------------------------------------------------------------------

def test_mark_invalid_zeros_replaces_only_targeted_columns():
	df = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 2.0]})
	out = mark_invalid_zeros_as_missing(df, ["a"])
	assert out["a"].isna().tolist() == [True, False]
	assert out["b"].tolist() == [0.0, 2.0]


def test_mark_invalid_zeros_does_not_mutate_input():
	df = pd.DataFrame({"a": [0.0, 1.0]})
	mark_invalid_zeros_as_missing(df, ["a"])
	assert df["a"].tolist() == [0.0, 1.0]


def test_mark_invalid_zeros_missing_column_raises():
	df = pd.DataFrame({"a": [1.0]})
	with pytest.raises(KeyError):
		mark_invalid_zeros_as_missing(df, ["nope"])


# ---------------------------------------------------------------------------
# clean_target
# ---------------------------------------------------------------------------

def test_clean_target_maps_yes_no_and_drops_nan():
	df = pd.DataFrame({"target": ["Yes", "No", None, "Yes"], "x": [1, 2, 3, 4]})
	cleaned = clean_target(df, TARGET)
	assert cleaned[TARGET].tolist() == [1, 0, 1]
	assert len(cleaned) == 3


def test_clean_target_keeps_numeric_binary():
	df = pd.DataFrame({"target": [0.0, 1.0, np.nan, 1.0], "x": [1, 2, 3, 4]})
	cleaned = clean_target(df, TARGET)
	assert cleaned[TARGET].tolist() == [0, 1, 1]
	assert cleaned[TARGET].dtype == int


# ---------------------------------------------------------------------------
# fit_preprocessor / transform_features
# ---------------------------------------------------------------------------

@pytest.fixture()
def fitted_result(raw_df, numeric_columns, categorical_columns):
	return fit_preprocessor(raw_df, numeric_columns, categorical_columns, target_column=TARGET)


def test_fit_returns_fitted_preprocessor_and_frame(fitted_result, raw_df):
	fitted, train_processed = fitted_result
	assert isinstance(fitted, FittedPreprocessor)
	# All rows survive; target first, then features.
	assert list(train_processed.columns)[0] == TARGET
	assert len(train_processed) == len(raw_df)


def test_numeric_imputation_uses_train_median(fitted_result, raw_df):
	fitted, train_processed = fitted_result
	expected_median = float(raw_df["num1"].median())  # median of non-NaN values
	assert fitted.numeric_imputer.statistics_[NUMERIC.index("num1")] == pytest.approx(expected_median)
	# No NaNs anywhere in the processed output.
	assert not train_processed.isna().any().any()


def test_categorical_imputation_uses_train_mode(fitted_result, raw_df):
	fitted, processed = fitted_result
	assert fitted.categorical_modes["bin1"] == raw_df["bin1"].mode().iloc[0]
	# multi1 got one-hot expanded; its dummies and the surviving binary column
	# must be NaN-free.
	surviving = [c for c in CATEGORICAL if c in processed.columns]
	assert not processed[surviving].isna().any().any()
	dummy_cols = [c for c in processed.columns if c.startswith("multi1_")]
	assert not processed[dummy_cols].isna().any().any()


def test_binary_encoding_is_deterministic_sorted_mapping(fitted_result):
	fitted, processed = fitted_result
	assert fitted.binary_maps["bin1"] == {"N": 0, "Y": 1}  # sorted order
	assert set(processed["bin1"].unique()) <= {0, 1}


def test_multi_category_one_hot_expands_columns(fitted_result):
	fitted, processed = fitted_result
	assert fitted.multi_columns == ["multi1"]
	assert {"multi1_a", "multi1_b", "multi1_c"} <= set(processed.columns)
	for column in ("multi1_a", "multi1_b", "multi1_c"):
		assert set(processed[column].unique()) <= {0, 1}
	# Original multi-category column is gone.
	assert "multi1" not in processed.columns


def test_scaling_zero_mean_unit_variance_on_train(fitted_result, raw_df):
	_, processed = fitted_result
	for column in NUMERIC:
		assert processed[column].mean() == pytest.approx(0.0, abs=1e-9)
		assert processed[column].std(ddof=0) == pytest.approx(1.0, abs=1e-9)


def test_output_feature_dimensions(fitted_result):
	"""Expected width: target + numeric + binary + one-hot dummies."""
	fitted, processed = fitted_result
	expected_width = 1 + len(NUMERIC) + len(fitted.binary_columns) + 3  # multi1 has 3 categories
	assert processed.shape == (10, expected_width)


def test_fit_does_not_mutate_input_frame(raw_df, numeric_columns, categorical_columns):
	before = raw_df.copy(deep=True)
	fit_preprocessor(raw_df, numeric_columns, categorical_columns, target_column=TARGET)
	pd.testing.assert_frame_equal(raw_df, before)


def test_refit_with_same_data_is_identical(raw_df, numeric_columns, categorical_columns):
	fitted_a, processed_a = fit_preprocessor(raw_df, numeric_columns, categorical_columns, TARGET)
	fitted_b, processed_b = fit_preprocessor(raw_df, numeric_columns, categorical_columns, TARGET)
	pd.testing.assert_frame_equal(processed_a, processed_b)
	assert np.allclose(fitted_a.scaler.mean_, fitted_b.scaler.mean_)
	assert np.allclose(fitted_a.numeric_imputer.statistics_, fitted_b.numeric_imputer.statistics_)


# ---------------------------------------------------------------------------
# transform_features: leakage-safe reuse of fitted statistics
# ---------------------------------------------------------------------------

def test_transform_applies_train_statistics_not_test_statistics():
	"""Scaler mean must come from train only: if transform re-fitted on the
	test split the means would differ."""
	train = pd.DataFrame(
		{
			"target": [0, 1] * 5,
			"num": [0.0] * 9 + [90.0],
			"cat": ["x"] * 10,
		}
	)
	test = pd.DataFrame({"target": [0, 1], "num": [100.0, 200.0], "cat": ["x", "x"]})
	fitted, _ = fit_preprocessor(train, ["num"], ["cat"], TARGET)
	transformed = transform_features(test, fitted, TARGET)
	# z-score of 100 under train stats (mean=9, std≈27.14), not test stats.
	expected = (100.0 - train["num"].mean()) / train["num"].std(ddof=0)
	assert transformed["num"].iloc[0] == pytest.approx(expected)


def test_transform_handles_unseen_category_via_fallback():
	train = pd.DataFrame({"target": [0, 1] * 5, "num": range(10), "cat": ["x"] * 10})
	test = pd.DataFrame({"target": [0], "num": [5], "cat": ["never_seen"]})
	fitted, _ = fit_preprocessor(train, ["num"], ["cat"], TARGET)
	transformed = transform_features(test, fitted, TARGET)
	# Single-category column is one-hot encoded; unseen value -> dummy stays 0,
	# never a raw string leaking into numeric model input.
	assert transformed["cat_x"].iloc[0] == 0
	assert transformed.select_dtypes(exclude="number").empty


def test_transform_aligns_missing_dummy_columns():
	train = pd.DataFrame({"target": [0, 1] * 5, "num": range(10), "cat": ["x"] * 10})
	test = pd.DataFrame({"target": [0], "num": [5], "cat": ["y"]})  # unseen category
	fitted, _ = fit_preprocessor(train, ["num"], ["cat"], TARGET)
	transformed = transform_features(test, fitted, TARGET)
	assert "cat_x" in transformed.columns
	assert transformed["cat_x"].iloc[0] == 0  # unseen category -> all dummies 0
	expected_columns = fit_preprocessor(train, ["num"], ["cat"], TARGET)[1].columns
	assert list(transformed.columns) == list(expected_columns)


def test_single_category_column_is_one_hot_encoded_not_passed_through():
	"""Regression: a categorical column with one training value used to slip
	through both encodings and reach transform as a raw string."""
	train = pd.DataFrame({"target": [0, 1] * 5, "num": range(10), "single": ["x"] * 10})
	fitted, processed = fit_preprocessor(train, ["num"], ["single"], TARGET)
	assert fitted.multi_columns == ["single"]
	assert "single_x" in processed.columns
	assert "single" not in processed.columns
	test = pd.DataFrame({"target": [0], "num": [3], "single": ["surprise"]})
	transformed = transform_features(test, fitted, TARGET)
	assert "single_x" in transformed.columns
	assert transformed.select_dtypes(exclude="number").empty


def test_transform_does_not_mutate_input_frame():
	test = pd.DataFrame({"target": [0], "num": [np.nan], "cat": ["x"]})
	fitted, _ = fit_preprocessor(
		pd.DataFrame({"target": [0, 1] * 5, "num": [float(i) for i in range(10)], "cat": ["x"] * 10}),
		["num"],
		["cat"],
		TARGET,
	)
	before = test.copy(deep=True)
	transform_features(test, fitted, TARGET)
	pd.testing.assert_frame_equal(test, before)


def test_transform_fills_missing_values_with_train_stats():
	train = pd.DataFrame({"target": [0, 1] * 5, "num": [float(i) for i in range(10)], "cat": ["x"] * 10})
	test = pd.DataFrame({"target": [0], "num": [np.nan], "cat": [None]})
	fitted, _ = fit_preprocessor(train, ["num"], ["cat"], TARGET)
	transformed = transform_features(test, fitted, TARGET)
	assert not transformed.isna().any().any()
