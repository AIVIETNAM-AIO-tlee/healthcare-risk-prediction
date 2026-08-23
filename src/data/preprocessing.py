from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from config import TARGET_COLUMN


def clean_target(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
	"""Drop rows with a missing target and encode it to a binary 0/1 integer.

	Mirrors the EDA finding that ~0.7% of rows have a missing target: since a
	label cannot be reliably imputed, those rows are dropped rather than filled.
	"""
	cleaned = df.dropna(subset=[target_column]).reset_index(drop=True)
	cleaned[target_column] = cleaned[target_column].map({"Yes": 1, "No": 0}).astype(int)
	return cleaned


@dataclass
class FittedPreprocessor:
	"""Bundle of everything fitted on the training split only.

	Passing this single object to ``transform_features`` guarantees the test
	split (and, later, any new query) goes through exactly the same imputers,
	encoders, and scaler that were fitted on the training data.
	"""

	numeric_columns: list[str]
	categorical_columns: list[str]
	binary_columns: list[str]
	multi_columns: list[str]
	numeric_imputer: SimpleImputer
	categorical_modes: pd.Series
	binary_maps: dict[str, dict[str, int]]
	binary_fallback: dict[str, int]
	dummy_columns: list[str]
	scaler: StandardScaler
	feature_order: list[str] = field(default_factory=list)


def fit_preprocessor(
	train_df: pd.DataFrame,
	numeric_columns: list[str],
	categorical_columns: list[str],
	target_column: str = TARGET_COLUMN,
) -> tuple[FittedPreprocessor, pd.DataFrame]:
	"""Fit imputers/encoders/scaler on the training split and transform it.

	Steps (same order as the EDA/preprocessing notebook):
	  1. Impute missing numeric values with the column median.
	  2. Impute missing categorical values with the column mode.
	  3. Label-encode binary (2-category) columns to 0/1.
	  4. One-hot encode multi-category columns.
	  5. Standardize numeric columns to zero mean / unit variance.
	"""
	df = train_df.copy()

	# 1) Impute numeric columns with the median (robust to outliers).
	numeric_imputer = SimpleImputer(strategy="median")
	df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

	# 2) Impute categorical columns with the mode (most frequent category).
	categorical_modes = df[categorical_columns].mode().iloc[0]
	df[categorical_columns] = df[categorical_columns].fillna(categorical_modes)

	# 3) Label-encode binary columns to 0/1, remembering the mapping and a
	#    fallback code (the training mode) for categories never seen at fit time.
	binary_columns = [c for c in categorical_columns if df[c].nunique() == 2]
	multi_columns = [c for c in categorical_columns if df[c].nunique() > 2]

	binary_maps: dict[str, dict[str, int]] = {}
	binary_fallback: dict[str, int] = {}
	for column in binary_columns:
		categories = sorted(df[column].unique())
		binary_maps[column] = {category: index for index, category in enumerate(categories)}
		df[column] = df[column].map(binary_maps[column])
		binary_fallback[column] = int(df[column].mode().iloc[0])

	# 4) One-hot encode multi-category columns. dtype=int keeps output as 0/1
	#    instead of pandas' default True/False bools, which write out much
	#    more compactly to CSV.
	df = pd.get_dummies(df, columns=multi_columns, drop_first=False, dtype=int)
	dummy_columns = [
		column
		for column in df.columns
		if column not in numeric_columns + binary_columns + [target_column]
	]

	# 5) Standardize numeric columns.
	scaler = StandardScaler()
	df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

	feature_order = numeric_columns + binary_columns + dummy_columns
	column_order = [target_column] + feature_order if target_column in df.columns else feature_order
	df = df[column_order]

	fitted = FittedPreprocessor(
		numeric_columns=numeric_columns,
		categorical_columns=categorical_columns,
		binary_columns=binary_columns,
		multi_columns=multi_columns,
		numeric_imputer=numeric_imputer,
		categorical_modes=categorical_modes,
		binary_maps=binary_maps,
		binary_fallback=binary_fallback,
		dummy_columns=dummy_columns,
		scaler=scaler,
		feature_order=feature_order,
	)
	return fitted, df


def transform_features(
	df: pd.DataFrame,
	fitted: FittedPreprocessor,
	target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
	"""Apply an already-fitted preprocessor to another split (e.g. the test set).

	Uses only ``fitted`` (learned from the training split) so no statistic from
	this split leaks into the transform -- new/unseen categories fall back to
	the training-time most-frequent value instead of raising an error.
	"""
	out = df.copy()

	out[fitted.numeric_columns] = fitted.numeric_imputer.transform(out[fitted.numeric_columns])
	out[fitted.categorical_columns] = out[fitted.categorical_columns].fillna(fitted.categorical_modes)

	for column in fitted.binary_columns:
		out[column] = out[column].map(fitted.binary_maps[column])
		out[column] = out[column].fillna(fitted.binary_fallback[column]).astype(int)

	out = pd.get_dummies(out, columns=fitted.multi_columns, drop_first=False, dtype=int)
	for column in fitted.dummy_columns:
		if column not in out.columns:
			out[column] = 0

	column_order = (
		[target_column] + fitted.feature_order if target_column in out.columns else fitted.feature_order
	)
	out = out[column_order]

	out[fitted.numeric_columns] = fitted.scaler.transform(out[fitted.numeric_columns])
	return out
