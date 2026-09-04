"""End-to-end experiment output tests (scope: 'Validate model results and
output formats' + 'Test preprocessing consistency across the three datasets').

Runs the real experiment runner against a temporary project (three numeric
train/test splits, the three real model configs, 3-fold CV) and pins the
resulting CSV/JSON contract: file set, columns, row granularity, and metric
value ranges. Keeps every artifact inside tmp_path so the repo's own results/
directory is never touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.run_models import run_experiments

DATASET_KEYS = ("dataset1", "dataset2", "dataset3")
MODEL_KEYS = ("adaboost", "xgboost", "lightgbm")
N_SPLITS = 3

FOLD_COLUMNS = {
    "dataset_key", "dataset_name", "model_key", "model_name", "fold",
    "n_train", "n_validation", "train_positive_rate", "validation_positive_rate",
    "fit_seconds", "roc_auc", "pr_auc", "recall", "f1",
}
METRIC_NAMES = ("roc_auc", "pr_auc", "recall", "f1")


def _write_split(train_path: Path, test_path: Path, seed: int) -> None:
    """Numeric binary-split CSVs with both classes guaranteed on both sides."""
    rng = np.random.default_rng(seed)
    n = 200
    y = (np.arange(n) % 7 == 0).astype(int)  # ~14% positives everywhere
    X = rng.normal(size=(n, 6)) + y[:, None] * 0.9
    columns = [f"f{i}" for i in range(X.shape[1])]
    cut = 150
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    for path, frame_X, frame_y in (
        (train_path, X[:cut], y[:cut]),
        (test_path, X[cut:], y[cut:]),
    ):
        pd.DataFrame(frame_X, columns=columns).assign(target=frame_y).to_csv(
            path, index=False
        )


@pytest.fixture(scope="module")
def experiment_outputs(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("experiment_project")
    with (Path(__file__).resolve().parents[2] / "config.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        repo_config = yaml.safe_load(stream)

    datasets = {}
    for index, key in enumerate(DATASET_KEYS, start=1):
        train_path = root / "data" / "processed" / key / "train.csv"
        test_path = root / "data" / "processed" / key / "test.csv"
        _write_split(train_path, test_path, seed=index)
        source = repo_config["datasets"][key]
        datasets[key] = {
            "name": source["name"],
            "train_path": str(train_path.relative_to(root)),
            "test_path": str(test_path.relative_to(root)),
            "target_column": "target",  # our synthetic splits name it 'target'
        }

    config = {
        "experiment": {
            "name": "qa_output_format",
            "random_state": 42,
            "decision_threshold": 0.50,
            "primary_metric": "pr_auc",
            "balance_training": True,
            "output_dir": "results/model_evaluation",
            "cv": {"n_splits": N_SPLITS, "shuffle": True},
        },
        "datasets": datasets,
        "models": repo_config["models"],
    }
    config_path = root / "config.yaml"
    with config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream)

    return run_experiments(config_path)


def test_all_expected_result_files_are_written(experiment_outputs):
    assert set(experiment_outputs) == {
        "fold_metrics", "cv_summary", "test_metrics",
        "model_comparison", "overall_model_comparison", "metadata",
    }
    assert all(path.exists() for path in experiment_outputs.values())


def test_fold_metrics_granularity_and_ranges(experiment_outputs):
    folds = pd.read_csv(experiment_outputs["fold_metrics"])

    assert set(FOLD_COLUMNS).issubset(folds.columns)
    assert len(folds) == len(DATASET_KEYS) * len(MODEL_KEYS) * N_SPLITS
    assert sorted(folds["fold"].unique()) == list(range(1, N_SPLITS + 1))
    assert sorted(folds["dataset_key"].unique()) == sorted(DATASET_KEYS)
    assert sorted(folds["model_key"].unique()) == sorted(MODEL_KEYS)
    for metric in METRIC_NAMES:
        assert folds[metric].between(0.0, 1.0).all()
    for rate_column in ("train_positive_rate", "validation_positive_rate"):
        assert folds[rate_column].between(0.0, 1.0).all()
    assert (folds["fit_seconds"] >= 0).all()


def test_cv_summary_one_row_per_dataset_model(experiment_outputs):
    summary = pd.read_csv(experiment_outputs["cv_summary"])

    assert len(summary) == len(DATASET_KEYS) * len(MODEL_KEYS)
    for metric in METRIC_NAMES:
        for stat in ("mean", "std", "min", "max", "range"):
            assert f"{metric}_{stat}" in summary.columns
        # range must equal max - min wherever the fold std is well-defined
        computed = summary[f"{metric}_max"] - summary[f"{metric}_min"]
        assert np.allclose(computed, summary[f"{metric}_range"])


def test_test_metrics_match_holdout_rows(experiment_outputs):
    test_frame = pd.read_csv(experiment_outputs["test_metrics"])

    assert len(test_frame) == len(DATASET_KEYS) * len(MODEL_KEYS)
    assert {"n_development", "n_test"}.issubset(test_frame.columns)
    assert (test_frame["n_test"] == 50).all()  # our splits: 150/50
    # hold-out rows keep the plain metric names; the test_* prefixes are
    # added later when merged into model_comparison.csv
    for metric in METRIC_NAMES:
        assert metric in test_frame.columns
        assert test_frame[metric].between(0.0, 1.0).all()


def test_model_comparison_joins_cv_and_test_views(experiment_outputs):
    comparison = pd.read_csv(experiment_outputs["model_comparison"])

    assert len(comparison) == len(DATASET_KEYS) * len(MODEL_KEYS)
    assert "cv_rank" in comparison.columns
    assert sorted(comparison["cv_rank"].unique()) == [1, 2, 3]
    for metric in METRIC_NAMES:
        assert f"test_{metric}" in comparison.columns


def test_overall_comparison_is_one_row_per_model(experiment_outputs):
    overall = pd.read_csv(experiment_outputs["overall_model_comparison"])

    assert sorted(overall["model_key"]) == sorted(MODEL_KEYS)
    assert "mean_dataset_rank" in overall.columns
    assert overall["mean_dataset_rank"].between(1, len(DATASET_KEYS)).all()


def test_metadata_is_valid_json(experiment_outputs):
    metadata = json.loads(experiment_outputs["metadata"].read_text(encoding="utf-8"))

    assert isinstance(metadata, dict)


def test_unknown_dataset_key_raises(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "random_state": 42,
                    "decision_threshold": 0.5,
                    "primary_metric": "pr_auc",
                    "balance_training": True,
                    "output_dir": "results/x",
                    "cv": {"n_splits": 2, "shuffle": True},
                },
                "datasets": {
                    "dataset1": {
                        "name": "d", "train_path": "a.csv",
                        "test_path": "b.csv", "target_column": "t",
                    }
                },
                "models": {"adaboost": {"display_name": "AdaBoost", "params": {}}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown dataset keys"):
        run_experiments(config_path, selected_datasets={"nope"})
