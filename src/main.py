from __future__ import annotations

from config import (
	NUMERIC_COLUMNS,
	PROCESSED_DIR,
	RANDOM_STATE,
	RAW_DATA_PATH,
	TARGET_COLUMN,
	TEST_CSV_PATH,
	TEST_SIZE,
	TRAIN_CSV_PATH,
)
from data.loader import infer_column_types, load_raw_dataset
from data.preprocessing import clean_target, fit_preprocessor, transform_features
from data.split import split_train_test


def main() -> None:
	print(f"Loading raw dataset from: {RAW_DATA_PATH}")
	df = load_raw_dataset(RAW_DATA_PATH)
	print(f"Raw shape: {df.shape}")

	df = clean_target(df, target_column=TARGET_COLUMN)
	print(f"Shape after dropping rows with missing target: {df.shape}")

	numeric_columns, categorical_columns = infer_column_types(
		df, NUMERIC_COLUMNS, target_column=TARGET_COLUMN
	)
	print(f"Numeric columns ({len(numeric_columns)}): {numeric_columns}")
	print(f"Categorical columns ({len(categorical_columns)}): {categorical_columns}")

	split_data = split_train_test(
		df, target_column=TARGET_COLUMN, test_size=TEST_SIZE, random_state=RANDOM_STATE
	)
	print(f"Train rows: {len(split_data.train_df)} | Test rows: {len(split_data.test_df)}")

	# Fit every imputer/encoder/scaler on the training split only, then reuse
	# those exact fitted objects to transform the test split (no data leakage).
	fitted, train_processed = fit_preprocessor(
		split_data.train_df, numeric_columns, categorical_columns, target_column=TARGET_COLUMN
	)
	test_processed = transform_features(split_data.test_df, fitted, target_column=TARGET_COLUMN)

	print(f"Processed train shape: {train_processed.shape}")
	print(f"Processed test shape: {test_processed.shape}")

	PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
	# 6 decimal places is far more precision than a standardized z-score needs
	# and keeps the CSV files a fraction of the size of full float64 output.
	train_processed.to_csv(TRAIN_CSV_PATH, index=False, float_format="%.6f")
	test_processed.to_csv(TEST_CSV_PATH, index=False, float_format="%.6f")
	print(f"Saved: {TRAIN_CSV_PATH}")
	print(f"Saved: {TEST_CSV_PATH}")
	print("Data processing complete.")


if __name__ == "__main__":
	main()
