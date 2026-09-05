"""Outlier handling tests (scope: 'Test outlier handling').

Covers the Tukey-fence IQR clipping policy: bounds are fit on the training
split only, applied unchanged to any split, clip instead of drop rows, and
stay NaN-safe so downstream imputation keeps working.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.preprocessing import apply_outlier_bounds, fit_outlier_bounds


def _train_frame() -> pd.DataFrame:
    # linear-interpolation quantiles: Q1 = 3.25, Q3 = 7.75 -> IQR = 4.5
    # -> default fence [3.25 - 6.75, 7.75 + 6.75] = [-3.5, 14.5]
    return pd.DataFrame({"value": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]})


def test_bounds_match_tukey_fence_formula():
    fitted = fit_outlier_bounds(_train_frame(), ["value"])

    assert fitted.bounds["value"] == (-3.5, 14.5)


def test_custom_multiplier_widens_bounds():
    fitted = fit_outlier_bounds(_train_frame(), ["value"], multiplier=3.0)

    # 3.25 +/- 3*4.5 and 7.75 +/- 3*4.5
    assert fitted.bounds["value"] == (-10.25, 21.25)


def test_fit_ignores_missing_values():
    frame = pd.DataFrame({"value": [np.nan, 1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

    fitted = fit_outlier_bounds(frame, ["value"])

    assert fitted.bounds["value"] == (-3.5, 14.5)


def test_apply_clips_values_into_fitted_bounds():
    fitted = fit_outlier_bounds(_train_frame(), ["value"])
    target = pd.DataFrame({"value": [-50.0, 5.0, 500.0]})

    clipped = apply_outlier_bounds(target, fitted)

    assert clipped["value"].tolist() == [-3.5, 5.0, 14.5]


def test_apply_preserves_row_count_and_missing_values():
    fitted = fit_outlier_bounds(_train_frame(), ["value"])
    target = pd.DataFrame({"value": [np.nan, -100.0, 100.0]})

    clipped = apply_outlier_bounds(target, fitted)

    assert len(clipped) == 3
    assert clipped["value"].isna().tolist() == [True, False, False]


def test_apply_does_not_mutate_input():
    fitted = fit_outlier_bounds(_train_frame(), ["value"])
    target = pd.DataFrame({"value": [-50.0, 5.0, 500.0]})

    apply_outlier_bounds(target, fitted)

    assert target["value"].tolist() == [-50.0, 5.0, 500.0]


def test_unfitted_columns_pass_through_untouched():
    fitted = fit_outlier_bounds(_train_frame(), ["value"])
    target = pd.DataFrame({"value": [-50.0, 500.0], "other": [-50.0, 500.0]})

    clipped = apply_outlier_bounds(target, fitted)

    assert clipped["other"].tolist() == [-50.0, 500.0]


def test_test_split_extremes_never_influence_fitted_bounds():
    """Leakage guard: bounds come from train only, so applying them to a test
    split with wilder extremes still clips to the train-derived fence."""
    train = _train_frame()
    test = pd.DataFrame({"value": [1.0, 5.0, 10.0, 10_000.0]})

    train_bounds = fit_outlier_bounds(train, ["value"])
    test_bounds = fit_outlier_bounds(test, ["value"])

    assert train_bounds.bounds["value"] != test_bounds.bounds["value"]

    clipped = apply_outlier_bounds(test, train_bounds)
    assert clipped["value"].max() <= train_bounds.bounds["value"][1]
