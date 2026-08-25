from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw_dataset(csv_path: str | Path, target_column: str) -> pd.DataFrame:
	"""Load a raw dataset CSV and do minimal, non-destructive normalization.

	This only strips column-name whitespace and drops columns that are
	entirely empty; it does not impute, encode, or scale anything -- that is
	the job of ``preprocessing.py``.
	"""
	path = Path(csv_path)
	if not path.exists():
		raise FileNotFoundError(
			f"Raw dataset not found: {path}. Make sure the dataset's CSV file has "
			f"been placed inside its 'data/raw/<dataset_key>/' folder."
		)

	df = pd.read_csv(path)
	df.columns = [str(column).strip() for column in df.columns]

	empty_columns = df.columns[df.isna().all()].tolist()
	if empty_columns:
		df = df.drop(columns=empty_columns)

	if target_column not in df.columns:
		raise ValueError(f"Expected target column '{target_column}', got {df.columns.tolist()}")

	return df


def infer_feature_columns(df: pd.DataFrame, target_column: str) -> list[str]:
	"""Return every column except the target column."""
	return [column for column in df.columns if column != target_column]


def infer_column_types(
	df: pd.DataFrame,
	numeric_columns: list[str],
	target_column: str,
) -> tuple[list[str], list[str]]:
	"""Split feature columns into (numeric, categorical) using a known numeric list.

	Any feature column not listed in ``numeric_columns`` is treated as
	categorical. This keeps the split explicit and reproducible instead of
	relying on pandas dtype inference, which is unreliable on columns that
	still contain missing values.
	"""
	feature_columns = infer_feature_columns(df, target_column=target_column)
	numeric = [column for column in feature_columns if column in numeric_columns]
	categorical = [column for column in feature_columns if column not in numeric_columns]
	return numeric, categorical
