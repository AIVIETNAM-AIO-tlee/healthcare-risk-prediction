"""Leakage Vector 2 — feature hygiene: identifiers and targets must never reach model features.

Plan refs: phase-02-data-layer.md Requirements (patient_id excluded from features,
loaders never expose target), phase-02 Success Criteria ("No feature column equals
or derives from patient_id").

RED contract: implemented in Phase 2. These tests fail (NotImplementedError) until
loaders exist, then permanently guard every loader via the slug registry.
"""

from __future__ import annotations

import pandas as pd

from tree_risk_stability.data.base import DatasetBundle, load_dataset

DATASET_SLUGS = [
    "synthetic",
    "heart",
    "breast_cancer",
    "pima",
    "framingham",
    "stroke",
]

# Column names that must never appear in X regardless of dataset.
FORBIDDEN_ID_TOKENS = ("id", "patient_id", "record_id")


def _assert_no_identifier_features(bundle: DatasetBundle) -> None:
    lowered = {c.lower() for c in bundle.X.columns}
    for token in FORBIDDEN_ID_TOKENS:
        assert not any(token == c or c.endswith(f"_{token}") for c in lowered), (
            f"identifier-like column leaked into features: {[c for c in bundle.X.columns if c.lower().endswith(token)]}"
        )


def _assert_target_not_in_features(bundle: DatasetBundle, target_name: str) -> None:
    assert target_name.lower() not in {c.lower() for c in bundle.X.columns}, (
        f"target column '{target_name}' leaked into features"
    )
    # Target values must not be recoverable as an exact copy of any single feature.
    y_vals = bundle.y.reset_index(drop=True)
    for col in bundle.X.columns:
        x_vals = bundle.X[col].reset_index(drop=True)
        if x_vals.dtype == object or y_vals.dtype == object:
            continue
        try:
            if len(y_vals) == len(x_vals) and (y_vals.astype(float) == x_vals.astype(float)).all():
                raise AssertionError(f"feature '{col}' is an exact copy of the target")
        except (TypeError, ValueError):
            continue


def test_bundle_shape_contract():
    """DatasetBundle carries disjoint X/y with aligned lengths and matching row counts."""
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    y = pd.Series([0, 1])
    bundle = DatasetBundle(X=X, y=y, task_type="binary", feature_names=["a", "b"])
    assert bundle.feature_names == list(X.columns)
    assert len(bundle.X) == len(bundle.y)


def test_all_registered_loaders_exclude_identifiers_and_target():
    """Every registered slug must pass hygiene checks against its real file.

    Skips per-slug when data/raw/ is absent (Phase 1 not yet executed) so the RED
    suite is runnable from a fresh clone; once Phase 1 lands this runs everywhere.
    """
    import pathlib

    raw_dir = pathlib.Path("data/raw")
    if not raw_dir.exists():
        import pytest

        pytest.skip("Phase 1 dataset acquisition not executed yet")
    for slug in DATASET_SLUGS:
        bundle = load_dataset(slug)
        _assert_no_identifier_features(bundle)
        _assert_target_not_in_features(bundle, bundle.metadata.get("target_column", "target"))
