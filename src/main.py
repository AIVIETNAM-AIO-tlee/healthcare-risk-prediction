from __future__ import annotations

import pandas as pd

from config import (
	DATASETS,
	FEATURE_CORRELATION_THRESHOLD,
	FEATURE_VARIANCE_THRESHOLD,
	IQR_MULTIPLIER,
	N_SPLITS,
	DatasetConfig,
)
from data.loader import infer_column_types, load_raw_dataset
from data.preprocessing import (
	apply_feature_selector,
	apply_outlier_bounds,
	clean_target,
	compute_class_weights,
	fit_feature_selector,
	fit_outlier_bounds,
	fit_preprocessor,
	mark_invalid_zeros_as_missing,
	transform_features,
)
from data.split import split_train_test, stratified_kfold_splits


def process_dataset(spec: DatasetConfig) -> None:
	print(f"\n=== {spec.name} ({spec.key}) ===")
	print(f"Loading raw dataset from: {spec.raw_path}")
	df = load_raw_dataset(spec.raw_path, target_column=spec.target_column)
	print(f"Raw shape: {df.shape}")

	if spec.drop_duplicate_rows:
		before = len(df)
		df = df.drop_duplicates().reset_index(drop=True)
		print(f"Dropped {before - len(df)} duplicate rows")

	if spec.zero_as_missing_columns:
		df = mark_invalid_zeros_as_missing(df, spec.zero_as_missing_columns)

	df = clean_target(df, target_column=spec.target_column)
	print(f"Shape after dropping rows with missing target: {df.shape}")

	# Class-imbalance policy (F1) is applied at model-fit time (balanced
	# sample_weight, see src/experiments/run_models.py + config.yaml's
	# balance_training flag); reported here only, not resampled.
	class_weights = compute_class_weights(df[spec.target_column])
	print(f"Class weights (balanced, informational only): {class_weights}")

	numeric_columns, categorical_columns = infer_column_types(
		df, spec.numeric_columns, target_column=spec.target_column
	)
	print(f"Numeric columns ({len(numeric_columns)}): {numeric_columns}")
	print(f"Categorical columns ({len(categorical_columns)}): {categorical_columns}")

	split_data = split_train_test(df, target_column=spec.target_column)
	print(f"Train rows: {len(split_data.train_df)} | Test rows: {len(split_data.test_df)}")

	train_df, test_df = split_data.train_df, split_data.test_df
	if spec.iqr_outlier_columns:
		# Outlier handling (F5): bounds fit on the training split only, then
		# applied identically to both splits.
		outlier_bounds = fit_outlier_bounds(
			train_df, spec.iqr_outlier_columns, multiplier=IQR_MULTIPLIER
		)
		train_df = apply_outlier_bounds(train_df, outlier_bounds)
		test_df = apply_outlier_bounds(test_df, outlier_bounds)
		print(f"Applied IQR outlier clipping to: {spec.iqr_outlier_columns}")

	# Fit every imputer/encoder/scaler on the training split only, then reuse
	# those exact fitted objects to transform the test split (no data leakage).
	fitted, train_processed = fit_preprocessor(
		train_df, numeric_columns, categorical_columns, target_column=spec.target_column
	)
	test_processed = transform_features(test_df, fitted, target_column=spec.target_column)

	# Feature reduction: drop near-constant and highly-correlated columns,
	# fit on the processed training features only.
	selector = fit_feature_selector(
		train_processed,
		target_column=spec.target_column,
		variance_threshold=FEATURE_VARIANCE_THRESHOLD,
		correlation_threshold=FEATURE_CORRELATION_THRESHOLD,
	)
	if selector.dropped_columns:
		print(
			f"Feature reduction dropped {len(selector.dropped_columns)} column(s): "
			f"{selector.dropped_low_variance} (low-variance), "
			f"{selector.dropped_correlated} (correlated)"
		)
	train_processed = apply_feature_selector(train_processed, selector, target_column=spec.target_column)
	test_processed = apply_feature_selector(test_processed, selector, target_column=spec.target_column)

	print(f"Processed train shape: {train_processed.shape}")
	print(f"Processed test shape: {test_processed.shape}")

	spec.processed_dir.mkdir(parents=True, exist_ok=True)
	# 6 decimal places is far more precision than a standardized z-score needs
	# and keeps the CSV files a fraction of the size of full float64 output.
	train_processed.to_csv(spec.train_csv_path, index=False, float_format="%.6f")
	test_processed.to_csv(spec.test_csv_path, index=False, float_format="%.6f")
	print(f"Saved: {spec.train_csv_path}")
	print(f"Saved: {spec.test_csv_path}")

	# Stratified K-Fold validation (train split only -- the held-out test
	# split above never participates in this). Every training row is
	# assigned to exactly one of N_SPLITS validation folds; the remaining
	# folds serve as that fold's training data (e.g. for 5 folds: 4/5 train,
	# 1/5 validation, rotated across all 5 folds). ``fold_assignment`` is
	# written row-for-row in the same order as ``train_processed`` /
	# ``train.csv``, so ``kfold_indices.csv`` can be loaded alongside
	# ``train.csv`` and joined purely by row position.
	folds = stratified_kfold_splits(train_processed, target_column=spec.target_column, n_splits=N_SPLITS)
	fold_assignment = pd.Series(index=train_processed.index, dtype=int, name="fold")
	for fold_number, (_, val_idx) in enumerate(folds):
		fold_assignment.iloc[val_idx] = fold_number
	fold_assignment.to_frame().to_csv(spec.kfold_indices_path, index=False)
	print(f"Saved: {spec.kfold_indices_path}")
	for fold_number, (train_idx, val_idx) in enumerate(folds):
		val_positive_rate = train_processed[spec.target_column].iloc[val_idx].mean() * 100
		print(
			f"  Fold {fold_number}: train={len(train_idx)}, val={len(val_idx)}, "
			f"val positive rate={val_positive_rate:.2f}%"
		)


def main() -> None:
	for spec in DATASETS:
		process_dataset(spec)
	print("\nData processing complete for all datasets.")


if __name__ == "__main__":
	main()
