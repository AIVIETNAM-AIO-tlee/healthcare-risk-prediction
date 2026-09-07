from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATASETS_BY_KEY, DatasetConfig
from src.data.loader import infer_column_types, load_raw_dataset
from src.data.preprocessing import (
    apply_feature_selector,
    apply_outlier_bounds,
    clean_target,
    fit_feature_selector,
    fit_outlier_bounds,
    fit_preprocessor,
    mark_invalid_zeros_as_missing,
    sanitize_feature_names,
    transform_features,
)
from src.data.split import split_train_test


def load_clean_raw_split(
    project_root: Path,
    dataset_key: str,
    random_state: int,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, DatasetConfig]:
    """Load and split a registered raw dataset before learned preprocessing."""
    try:
        spec = DATASETS_BY_KEY[dataset_key]
    except KeyError as error:
        raise KeyError(f"No raw dataset metadata is registered for '{dataset_key}'.") from error

    raw_path = project_root / "data" / "raw" / dataset_key / spec.raw_filename
    frame = load_raw_dataset(raw_path, target_column=spec.target_column)
    if spec.drop_duplicate_rows:
        frame = frame.drop_duplicates().reset_index(drop=True)
    if spec.zero_as_missing_columns:
        frame = mark_invalid_zeros_as_missing(frame, spec.zero_as_missing_columns)
    frame = clean_target(frame, target_column=spec.target_column)
    split = split_train_test(
        frame,
        target_column=spec.target_column,
        test_size=test_size,
        random_state=random_state,
    )
    return split.train_df, split.test_df, spec


def preprocess_fold(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    spec: DatasetConfig,
    preprocessing_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every learned transform on fold-training rows only."""
    numeric_columns, categorical_columns = infer_column_types(
        train_df, spec.numeric_columns, target_column=spec.target_column
    )

    fitted_bounds = fit_outlier_bounds(
        train_df,
        spec.iqr_outlier_columns,
        multiplier=float(preprocessing_config["iqr_multiplier"]),
    ) if spec.iqr_outlier_columns else None
    if fitted_bounds is not None:
        train_df = apply_outlier_bounds(train_df, fitted_bounds)
        validation_df = apply_outlier_bounds(validation_df, fitted_bounds)

    fitted, train_processed = fit_preprocessor(
        train_df,
        numeric_columns,
        categorical_columns,
        target_column=spec.target_column,
    )
    validation_processed = transform_features(
        validation_df,
        fitted,
        target_column=spec.target_column,
    )
    selector = fit_feature_selector(
        train_processed,
        target_column=spec.target_column,
        variance_threshold=float(preprocessing_config["feature_variance_threshold"]),
        correlation_threshold=float(preprocessing_config["feature_correlation_threshold"]),
    )
    return (
        sanitize_feature_names(
            apply_feature_selector(train_processed, selector, target_column=spec.target_column),
            spec.target_column,
        ),
        sanitize_feature_names(
            apply_feature_selector(validation_processed, selector, target_column=spec.target_column),
            spec.target_column,
        ),
    )


def preprocess_full_split(
    train_df: pd.DataFrame,
    other_df: pd.DataFrame,
    spec: DatasetConfig,
    preprocessing_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit preprocessing on one complete split and transform another split."""
    return preprocess_fold(train_df, other_df, spec, preprocessing_config)
