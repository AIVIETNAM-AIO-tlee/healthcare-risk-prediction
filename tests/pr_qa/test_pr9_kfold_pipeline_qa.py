"""QA tests for PR #9 (preprocessing pipeline + persisted k-fold indices).

Targets the PR's changes without modifying them:
- src/config.py: new IQR / feature-reduction constants (must stay aligned
  with the preprocessing defaults this repo's suite already pins) and the
  DatasetConfig.kfold_indices_path seam
- src/main.py process_dataset: kfold_indices.csv must be row-aligned with
  train.csv, integral-typed, complete over folds, and stratified
- shipped data/processed/<dataset>/kfold_indices.csv artifacts

Self-skips when the PR code/artifacts are absent (e.g. on main).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _pr9_config():
    try:
        import config
    except ImportError:
        pytest.skip("src/config.py not importable on this branch")
    if not hasattr(config, "IQR_MULTIPLIER"):
        pytest.skip("PR #9 config constants not present on this branch")
    return config


def _pr9_main():
    _pr9_config()
    try:
        import main
    except ImportError:
        pytest.skip("PR #9 src/main.py not importable on this branch")
    return main


# ---------------------------------------------------------------------------
# Config constants stay aligned with the tested preprocessing defaults
# ---------------------------------------------------------------------------
def test_constants_match_preprocessing_function_defaults():
    from data import preprocessing

    config = _pr9_config()
    # older snapshots (e.g. PR #8's base) predate these functions
    if not (
        hasattr(preprocessing, "fit_outlier_bounds")
        and hasattr(preprocessing, "fit_feature_selector")
    ):
        pytest.skip("preprocessing functions not present on this snapshot")

    outlier_default = inspect.signature(preprocessing.fit_outlier_bounds).parameters["multiplier"].default
    variance_default = inspect.signature(preprocessing.fit_feature_selector).parameters["variance_threshold"].default
    correlation_default = inspect.signature(preprocessing.fit_feature_selector).parameters["correlation_threshold"].default

    assert config.IQR_MULTIPLIER == outlier_default == 1.5
    assert config.FEATURE_VARIANCE_THRESHOLD == variance_default == 1e-4
    assert config.FEATURE_CORRELATION_THRESHOLD == correlation_default == 0.9


def test_dataset_config_kfold_indices_path_seam(tmp_path):
    config = _pr9_config()

    spec = config.DatasetConfig(
        key="qa", name="QA", source_url="", raw_filename="qa.csv",
        target_column="target", numeric_columns=["num1"],
    )

    assert spec.kfold_indices_path == spec.processed_dir / "kfold_indices.csv"


# ---------------------------------------------------------------------------
# process_dataset end-to-end on a tiny synthetic dataset
# ---------------------------------------------------------------------------
def _write_tiny_raw_csv(raw_dir: Path, n: int = 150) -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(n, 3)), columns=["num1", "num2", "num3"])
    frame["smoker"] = np.where(rng.random(n) < 0.5, "Y", "N")
    frame["target"] = (frame.num1 > 0).astype(int)  # ~50% both classes solid
    raw_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(raw_dir / "qa.csv", index=False)


def _qa_spec(config, tmp_path: Path):
    return config.DatasetConfig(
        key="qa", name="QA Dataset", source_url="", raw_filename="qa.csv",
        target_column="target", numeric_columns=["num1", "num2", "num3"],
    )


def test_fit_preprocessor_crashes_on_zero_categorical_columns():
    """Documents a robustness gap (pre-existing, visible via main.py wiring):
    with an empty categorical list, ``df[[]].mode().iloc[0]`` raises
    IndexError instead of being a no-op."""
    from data.preprocessing import fit_preprocessor

    frame = pd.DataFrame({"num1": [1.0, 2.0, 3.0], "target": [0, 1, 0]})

    with pytest.raises(IndexError):
        fit_preprocessor(frame, ["num1"], [], target_column="target")


def test_process_dataset_writes_aligned_integral_kfold_indices(tmp_path, monkeypatch):
    config = _pr9_config()
    main_module = _pr9_main()
    monkeypatch.setattr(config, "RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    _write_tiny_raw_csv(tmp_path / "raw" / "qa")
    spec = _qa_spec(config, tmp_path)

    main_module.process_dataset(spec)

    train = pd.read_csv(spec.train_csv_path)
    kfold = pd.read_csv(spec.kfold_indices_path)
    assert list(kfold.columns) == ["fold"]
    assert len(kfold) == len(train)  # row-for-row alignment with train.csv
    assert kfold["fold"].dtype.kind == "i", "fold labels must be integral, not float"
    assert set(kfold["fold"].unique()) == set(range(config.N_SPLITS))
    assert kfold["fold"].notna().all()

    sizes = kfold["fold"].value_counts()
    assert (sizes <= np.ceil(len(kfold) / config.N_SPLITS)).all()
    assert (sizes >= np.floor(len(kfold) / config.N_SPLITS)).all()


def test_process_dataset_kfold_stratification(tmp_path, monkeypatch):
    config = _pr9_config()
    main_module = _pr9_main()
    monkeypatch.setattr(config, "RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    _write_tiny_raw_csv(tmp_path / "raw" / "qa")
    spec = _qa_spec(config, tmp_path)

    main_module.process_dataset(spec)

    train = pd.read_csv(spec.train_csv_path)
    kfold = pd.read_csv(spec.kfold_indices_path)
    overall_rate = train["target"].mean()
    # tiny fixture: one flipped row moves a fold rate by ~1/30 — 0.05 is the
    # granularity floor; the shipped-artifact test keeps the tight 0.01 bar
    for fold, rows in train.groupby(kfold["fold"].to_numpy()):
        assert abs(rows["target"].mean() - overall_rate) < 0.05, (
            f"fold {fold} positive rate drifted from overall (stratification broken)"
        )


def test_process_dataset_kfold_reproducible_across_runs(tmp_path, monkeypatch):
    config = _pr9_config()
    main_module = _pr9_main()
    monkeypatch.setattr(config, "RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    _write_tiny_raw_csv(tmp_path / "raw" / "qa")
    spec = _qa_spec(config, tmp_path)

    main_module.process_dataset(spec)
    first = spec.kfold_indices_path.read_bytes()
    main_module.process_dataset(spec)
    second = spec.kfold_indices_path.read_bytes()

    assert first == second


# ---------------------------------------------------------------------------
# Shipped artifacts in the PR tree
# ---------------------------------------------------------------------------
def test_shipped_kfold_indices_aligned_with_train_csv():
    shipped = [
        path.parent
        for path in (REPO_ROOT / "data" / "processed").glob("*/kfold_indices.csv")
    ]
    if not shipped:
        pytest.skip("PR #9 shipped kfold_indices.csv not present on this branch")

    for processed_dir in sorted(shipped):
        train = pd.read_csv(processed_dir / "train.csv")
        kfold = pd.read_csv(processed_dir / "kfold_indices.csv")

        assert list(kfold.columns) == ["fold"], processed_dir
        assert len(kfold) == len(train), (
            f"{processed_dir}: kfold rows must align row-for-row with train.csv"
        )
        assert kfold["fold"].dtype.kind == "i", (
            f"{processed_dir}: fold labels stored as float — must be integral"
        )
        assert set(kfold["fold"].unique()) == {0, 1, 2, 3, 4}
        assert kfold["fold"].notna().all()


def test_shipped_kfold_indices_are_stratified():
    processed_root = REPO_ROOT / "data" / "processed"
    datasets = sorted(processed_root.glob("*/kfold_indices.csv"))
    if not datasets:
        pytest.skip("PR #9 shipped kfold_indices.csv not present on this branch")

    for kfold_path in datasets:
        processed_dir = kfold_path.parent
        train = pd.read_csv(processed_dir / "train.csv")
        target = train[train.columns[0]]  # main.py writes the target first
        fold = pd.read_csv(kfold_path)["fold"]
        overall_rate = target.mean()

        rates = target.groupby(fold).mean()
        assert (rates - overall_rate).abs().max() < 0.01, (
            f"{processed_dir}: per-fold positive rates {rates.tolist()} "
            f"drift from overall {overall_rate:.4f}"
        )
