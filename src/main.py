from __future__ import annotations

from config import DATASETS, DatasetConfig
from data.loader import infer_column_types, load_raw_dataset
from data.preprocessing import (
	clean_target,
	fit_preprocessor,
	mark_invalid_zeros_as_missing,
	transform_features,
)
from data.split import split_train_test


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

	numeric_columns, categorical_columns = infer_column_types(
		df, spec.numeric_columns, target_column=spec.target_column
	)
	print(f"Numeric columns ({len(numeric_columns)}): {numeric_columns}")
	print(f"Categorical columns ({len(categorical_columns)}): {categorical_columns}")

	split_data = split_train_test(df, target_column=spec.target_column)
	print(f"Train rows: {len(split_data.train_df)} | Test rows: {len(split_data.test_df)}")

	# Fit every imputer/encoder/scaler on the training split only, then reuse
	# those exact fitted objects to transform the test split (no data leakage).
	fitted, train_processed = fit_preprocessor(
		split_data.train_df, numeric_columns, categorical_columns, target_column=spec.target_column
	)
	test_processed = transform_features(split_data.test_df, fitted, target_column=spec.target_column)

	print(f"Processed train shape: {train_processed.shape}")
	print(f"Processed test shape: {test_processed.shape}")

	spec.processed_dir.mkdir(parents=True, exist_ok=True)
	# 6 decimal places is far more precision than a standardized z-score needs
	# and keeps the CSV files a fraction of the size of full float64 output.
	train_processed.to_csv(spec.train_csv_path, index=False, float_format="%.6f")
	test_processed.to_csv(spec.test_csv_path, index=False, float_format="%.6f")
	print(f"Saved: {spec.train_csv_path}")
	print(f"Saved: {spec.test_csv_path}")


def main() -> None:
	for spec in DATASETS:
		process_dataset(spec)
	print("\nData processing complete for all datasets.")


if __name__ == "__main__":
	main()
