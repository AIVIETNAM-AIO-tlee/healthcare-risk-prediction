from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


def mark_invalid_zeros_as_missing(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
	"""Replace 0 with NaN in the given numeric columns.

	Some numeric health measurements (e.g. ``Cholesterol``/``RestingBP`` in
	dataset2) use 0 as a placeholder for "not measured" rather than a genuine
	reading. Converting those to NaN lets the normal median imputation in
	``fit_preprocessor``/``transform_features`` handle them, instead of the
	model seeing physiologically impossible zero values.
	"""
	out = df.copy()
	for column in columns:
		out[column] = out[column].replace(0, np.nan)
	return out


def clean_target(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
	"""Drop rows with a missing target and encode it to a binary 0/1 integer.

	Handles both a raw "Yes"/"No" string target (dataset1) and datasets whose
	target is already numeric 0/1 (dataset2, dataset3) -- either way, rows
	with a missing target are dropped first, since a label cannot be
	reliably imputed.

	The "is it a string column?" check uses ``is_numeric_dtype`` rather than
	comparing ``dtype == object``: pandas >= 2.x (and the ``str`` dtype used
	by default in pandas 3.x) can read text columns with a ``StringDtype``
	that is not ``object``, so an ``== object`` check silently misses them
	and this method must be dtype-version agnostic to catch both.
	"""
	cleaned = df.dropna(subset=[target_column]).reset_index(drop=True)
	if not pd.api.types.is_numeric_dtype(cleaned[target_column]):
		cleaned[target_column] = cleaned[target_column].map({"Yes": 1, "No": 0})
	cleaned[target_column] = cleaned[target_column].astype(int)
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
	target_column: str,
) -> tuple[FittedPreprocessor, pd.DataFrame]:
	"""Fit imputers/encoders/scaler on the training split and transform it.

	Steps (same order as the EDA/preprocessing notebooks):
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
	#    Single-category columns go through one-hot (step 4) instead: a raw
	#    string column would survive transform unchanged and poison model input.
	binary_columns = [c for c in categorical_columns if df[c].nunique() == 2]
	multi_columns = [c for c in categorical_columns if df[c].nunique() != 2]

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
	target_column: str,
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


# ---------------------------------------------------------------------------
# Class imbalance (reporting only -- see docs/qa-scope-methodology-review-
# handoff.md, finding F1)
# ---------------------------------------------------------------------------
def compute_class_weights(y: pd.Series) -> dict[int, float]:
	"""Compute sklearn "balanced" class weights for a binary target.

	These are the same per-class weights that
	``sklearn.utils.class_weight.compute_sample_weight(class_weight="balanced")``
	derives at model-fit time in ``src/experiments/run_models.py`` (see
	``config.yaml``'s ``balance_training`` flag) -- that is the project's one
	chosen class-imbalance policy (F1). This helper exposes the same numbers
	here purely for EDA/reporting (e.g. printing or plotting how imbalanced a
	dataset is, and what weight each class would receive). Preprocessing
	itself deliberately does not resample or re-weight rows: doing so here
	*in addition to* the balanced sample weights already applied at training
	time would double-correct for the same imbalance.
	"""
	classes = np.unique(y)
	weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
	return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


# ---------------------------------------------------------------------------
# Outlier handling (IQR / Tukey fences) -- see docs/qa-scope-methodology-
# review-handoff.md, finding F5
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FittedOutlierBounds:
	"""Per-column [lower, upper] IQR clipping bounds, fit on the training
	split only."""

	bounds: dict[str, tuple[float, float]]


def fit_outlier_bounds(
	train_df: pd.DataFrame,
	columns: list[str],
	multiplier: float = 1.5,
) -> FittedOutlierBounds:
	"""Compute Tukey IQR clipping bounds per column from the training split.

	``bounds = [Q1 - multiplier*IQR, Q3 + multiplier*IQR]``, the classic
	Tukey fence. Quantiles ignore NaNs, so this is safe to call before
	missing-value imputation (e.g. after ``mark_invalid_zeros_as_missing``).

	Only meant for genuinely continuous numeric columns on datasets where a
	quality reviewer has judged IQR clipping appropriate -- currently
	``DatasetConfig.iqr_outlier_columns``, empty by default. Mostly-binary or
	discrete survey data should not go through this: a genuinely high value
	(e.g. a high BMI) is real disease signal, not noise, and clipping it
	would destroy that signal.
	"""
	bounds: dict[str, tuple[float, float]] = {}
	for column in columns:
		q1 = train_df[column].quantile(0.25)
		q3 = train_df[column].quantile(0.75)
		iqr = q3 - q1
		lower = float(q1 - multiplier * iqr)
		upper = float(q3 + multiplier * iqr)
		bounds[column] = (lower, upper)
	return FittedOutlierBounds(bounds=bounds)


def apply_outlier_bounds(df: pd.DataFrame, fitted: FittedOutlierBounds) -> pd.DataFrame:
	"""Clip each fitted column to its [lower, upper] range.

	Values are clipped in place of the original, never dropped, so row
	counts and any external row alignment are preserved. Missing values are
	left as NaN (``Series.clip`` is NaN-safe), so downstream median
	imputation still handles them normally. Always uses ``fitted`` bounds
	computed on the training split, so a split's own extreme values never
	influence the bounds applied to it -- avoids leaking test-set
	information into preprocessing.
	"""
	out = df.copy()
	for column, (lower, upper) in fitted.bounds.items():
		out[column] = out[column].clip(lower=lower, upper=upper)
	return out


# ---------------------------------------------------------------------------
# Feature reduction (fit on train only, applied identically to test)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FittedFeatureSelector:
	"""Which processed feature columns to keep/drop, fit on the training
	split's already-encoded/scaled features only."""

	kept_columns: list[str]
	dropped_low_variance: list[str]
	dropped_correlated: list[str]

	@property
	def dropped_columns(self) -> list[str]:
		return self.dropped_low_variance + self.dropped_correlated


def fit_feature_selector(
	train_processed: pd.DataFrame,
	target_column: str,
	variance_threshold: float = 1e-4,
	correlation_threshold: float = 0.9,
) -> FittedFeatureSelector:
	"""Fit a lightweight feature-reduction selector on already-encoded/scaled
	training features (i.e. ``fit_preprocessor``'s output).

	Two simple, leakage-safe hygiene passes, both computed on the training
	split only and then applied identically to the test split via
	``apply_feature_selector``:

	  1. Drop near-constant columns (variance below ``variance_threshold``)
		 -- e.g. a one-hot dummy for a category so rare it carries almost no
		 signal. A properly standardized numeric column always has variance
		 ~1, so this step effectively only prunes binary/dummy columns.
	  2. Among the remaining columns, for every pair whose absolute Pearson
		 correlation exceeds ``correlation_threshold``, drop the second
		 column of the pair -- redundant, highly collinear features add
		 noise/dimensionality without adding information.

	The target column is always excluded from both checks and kept.
	"""
	feature_columns = [c for c in train_processed.columns if c != target_column]

	variances = train_processed[feature_columns].var()
	dropped_low_variance = variances[variances < variance_threshold].index.tolist()

	remaining = [c for c in feature_columns if c not in dropped_low_variance]
	dropped_correlated: list[str] = []
	if len(remaining) > 1:
		corr = train_processed[remaining].corr().abs()
		for i, col_i in enumerate(remaining):
			if col_i in dropped_correlated:
				continue
			for col_j in remaining[i + 1 :]:
				if col_j in dropped_correlated:
					continue
				if corr.loc[col_i, col_j] > correlation_threshold:
					dropped_correlated.append(col_j)

	kept_columns = [
		column
		for column in feature_columns
		if column not in dropped_low_variance and column not in dropped_correlated
	]
	return FittedFeatureSelector(
		kept_columns=kept_columns,
		dropped_low_variance=dropped_low_variance,
		dropped_correlated=dropped_correlated,
	)


def apply_feature_selector(
	df: pd.DataFrame,
	fitted: FittedFeatureSelector,
	target_column: str,
) -> pd.DataFrame:
	"""Keep only ``fitted.kept_columns`` (plus the target, if present), in order."""
	columns = ([target_column] if target_column in df.columns else []) + fitted.kept_columns
	return df[columns]
