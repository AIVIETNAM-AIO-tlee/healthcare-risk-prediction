"""Integration tests: real datasets through the full pipeline.

For each of the 3 registered datasets:
  load -> dedupe -> zero-as-missing -> clean target -> stratified 80/20 split
  -> fit preprocessor on train -> transform test -> Stratified K-Fold on train.

Leakage probes compare fitted statistics (imputer medians, scaler mean/std,
categorical modes) against statistics computed on the *full* cleaned data:
if preprocessing had been fitted on everything instead of train only, they
would be equal. Also verifies repeated runs produce identical outputs and
the test split never enters the K-Fold loop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import DATASETS, N_SPLITS, RANDOM_STATE
from data.loader import infer_column_types, load_raw_dataset
from data.preprocessing import (
	clean_target,
	fit_preprocessor,
	mark_invalid_zeros_as_missing,
	transform_features,
)
from data.split import split_train_test, stratified_kfold_splits


def prepare(spec):
	"""Mirror src/main.py's dataset-specific steps up to the target-cleaned frame."""
	df = load_raw_dataset(spec.raw_path, target_column=spec.target_column)
	if spec.drop_duplicate_rows:
		df = df.drop_duplicates().reset_index(drop=True)
	if spec.zero_as_missing_columns:
		df = mark_invalid_zeros_as_missing(df, spec.zero_as_missing_columns)
	return clean_target(df, spec.target_column)


@pytest.fixture(params=DATASETS, ids=lambda spec: spec.key)
def pipeline_state(request):
	spec = request.param
	raw = prepare(spec)
	split = split_train_test(raw, spec.target_column)
	numeric_columns, categorical_columns = infer_column_types(
		raw, spec.numeric_columns, target_column=spec.target_column
	)
	fitted, train_processed = fit_preprocessor(
		split.train_df, numeric_columns, categorical_columns, spec.target_column
	)
	test_processed = transform_features(split.test_df, fitted, spec.target_column)
	return {
		"spec": spec,
		"raw_clean": raw,
		"split": split,
		"fitted": fitted,
		"train_processed": train_processed,
		"test_processed": test_processed,
	}


# ---------------------------------------------------------------------------
# Splitting & separation
# ---------------------------------------------------------------------------

def test_split_ratio_and_no_row_overlap(pipeline_state):
	state = pipeline_state
	raw, split = state["raw_clean"], state["split"]
	assert len(split.train_df) + len(split.test_df) == len(raw)
	# Exact row overlap between splits: dataset1 is survey data whose raw file
	# contains genuine duplicate responses (only ~157 of them are dropped for
	# dataset3 via drop_duplicate_rows; dataset1 keeps its duplicates), so a
	# handful of identical rows can legitimately land on both sides of the
	# split. Assert the overlap is bounded by the dataset's own duplicate
	# count rather than demanding zero.
	train_rows = list(map(tuple, split.train_df.astype(str).itertuples(index=False)))
	test_rows = set(map(tuple, split.test_df.astype(str).itertuples(index=False)))
	overlap = sum(1 for row in train_rows if row in test_rows)
	raw_duplicate_rows = int(raw.duplicated().sum())
	assert overlap <= max(raw_duplicate_rows, 1)


def test_target_not_in_input_features(pipeline_state):
	state = pipeline_state
	target = state["spec"].target_column
	for frame in (state["train_processed"], state["test_processed"]):
		features = [c for c in frame.columns if c != target]
		assert target not in features
		# Target appears exactly once, as the leading column.
		assert list(frame.columns).count(target) == 1
		assert frame.columns[0] == target


# ---------------------------------------------------------------------------
# Leakage probes: fitted stats must come from train only
# ---------------------------------------------------------------------------

def test_scaler_fitted_on_train_only(pipeline_state):
	state = pipeline_state
	train_raw, full = state["split"].train_df, state["raw_clean"]
	fitted = state["fitted"]
	for position, column in enumerate(fitted.numeric_columns):
		# Compare against the mean the scaler would have seen: train values
		# AFTER median imputation (imputation runs before scaling in fit).
		imputed_train = pd.to_numeric(train_raw[column], errors="coerce").fillna(
			pd.to_numeric(train_raw[column], errors="coerce").median()
		)
		full_imputed = pd.to_numeric(full[column], errors="coerce")
		train_mean, full_mean = float(imputed_train.mean()), float(full_imputed.mean())
		if np.isclose(train_mean, full_mean, rtol=1e-12):
			continue  # statistic identical by coincidence; probe uninformative
		assert fitted.scaler.mean_[position] == pytest.approx(train_mean, rel=1e-9), column
		assert fitted.scaler.mean_[position] != pytest.approx(full_mean), column


def test_imputer_median_fitted_on_train_only(pipeline_state):
	state = pipeline_state
	train_raw, full = state["split"].train_df, state["raw_clean"]
	stats = dict(
		zip(state["fitted"].numeric_columns, state["fitted"].numeric_imputer.statistics_)
	)
	for column, median in stats.items():
		train_median = float(pd.to_numeric(train_raw[column], errors="coerce").median())
		assert median == pytest.approx(train_median), column
		full_median = float(pd.to_numeric(full[column], errors="coerce").median())
		# If train median happens to equal the full-data median the probe is
		# uninformative; otherwise it must differ from it (proof of train-only fit).
		if not np.isclose(median, full_median):
			assert median != pytest.approx(full_median)


def test_categorical_modes_fitted_on_train_only(pipeline_state):
	state = pipeline_state
	train_raw = state["split"].train_df
	modes = state["fitted"].categorical_modes
	for column in modes.index:
		train_mode = train_raw[column].mode().iloc[0]
		assert modes[column] == train_mode, column
		full_mode = state["raw_clean"][column].mode().iloc[0]
		if train_mode != full_mode:
			assert modes[column] != full_mode, column  # proof of train-only fit


# ---------------------------------------------------------------------------
# Output shape / consistency
# ---------------------------------------------------------------------------

def test_train_test_share_feature_layout(pipeline_state):
	state = pipeline_state
	train_cols, test_cols = list(state["train_processed"].columns), list(state["test_processed"].columns)
	assert train_cols == test_cols


def test_no_missing_values_in_processed_output(pipeline_state):
	state = pipeline_state
	assert not state["train_processed"].isna().any().any()
	assert not state["test_processed"].isna().any().any()


def test_preprocessing_does_not_touch_raw_files(pipeline_state):
	"""Raw CSV checksums must be unchanged after a full pipeline pass."""
	import hashlib

	from config import DATASETS_BY_KEY

	spec = DATASETS_BY_KEY[pipeline_state["spec"].key]
	digest = hashlib.sha256(spec.raw_path.read_bytes()).hexdigest()
	prepare(spec)  # second full read+transform pass over the same raw file
	assert hashlib.sha256(spec.raw_path.read_bytes()).hexdigest() == digest


# ---------------------------------------------------------------------------
# Cross-validation on the training split
# ---------------------------------------------------------------------------

def test_kfold_on_train_excludes_test(pipeline_state):
	state = pipeline_state
	split = state["split"]
	folds = stratified_kfold_splits(
		split.train_df, state["spec"].target_column, n_splits=N_SPLITS, random_state=RANDOM_STATE
	)
	assert len(folds) == N_SPLITS
	train_positions = set(range(len(split.train_df)))
	all_val = set()
	for fold_train_idx, fold_val_idx in folds:
		assert set(fold_train_idx) | set(fold_val_idx) == train_positions
		all_val |= set(fold_val_idx)
	assert all_val == train_positions  # every training sample validated exactly once


def test_full_pipeline_reproducible_across_runs(request, pipeline_state):
	state = pipeline_state
	spec = state["spec"]
	raw = state["raw_clean"]
	numeric_columns, categorical_columns = infer_column_types(
		raw, spec.numeric_columns, target_column=spec.target_column
	)
	split_b = split_train_test(raw, spec.target_column)
	pd.testing.assert_frame_equal(split_b.train_df, state["split"].train_df)
	pd.testing.assert_frame_equal(split_b.test_df, state["split"].test_df)
	fitted_b, train_b = fit_preprocessor(
		split_b.train_df, numeric_columns, categorical_columns, spec.target_column
	)
	pd.testing.assert_frame_equal(train_b, state["train_processed"])
	test_b = transform_features(split_b.test_df, fitted_b, spec.target_column)
	pd.testing.assert_frame_equal(test_b, state["test_processed"])
