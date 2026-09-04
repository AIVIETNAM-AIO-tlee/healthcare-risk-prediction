from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from src.evaluation.metrics import (
    METRIC_NAMES,
    add_dataset_ranks,
    build_overall_comparison,
    compute_binary_metrics,
    summarize_fold_metrics,
)
from src.evaluation.shap_stability import compute_fold_shap_importance, summarize_shap_stability
from src.experiment_config import load_experiment_config
from src.models.factory import build_model


def _is_git_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as stream:
        prefix = stream.read(128)
    return prefix.startswith(b"version https://git-lfs.github.com/spec/v1")


def _assert_real_csv(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if _is_git_lfs_pointer(path):
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not the actual CSV data. "
            "Run 'git lfs install' and 'git lfs pull' in a full repository clone, "
            "then rerun the experiment."
        )


def _load_processed_split(
    project_root: Path,
    dataset_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_path = project_root / dataset_config["train_path"]
    test_path = project_root / dataset_config["test_path"]
    _assert_real_csv(train_path)
    _assert_real_csv(test_path)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    target = dataset_config["target_column"]

    for split_name, frame in (("train", train_df), ("test", test_df)):
        if target not in frame.columns:
            raise ValueError(
                f"Target column '{target}' is missing from {split_name} split: "
                f"{dataset_config['name']}"
            )
        if frame[target].isna().any():
            raise ValueError(f"Target column contains missing values in {split_name} split.")

    feature_columns = [column for column in train_df.columns if column != target]
    if feature_columns != [column for column in test_df.columns if column != target]:
        raise ValueError("Train/test feature columns or their order do not match.")

    X_train = train_df[feature_columns].copy()
    y_train = train_df[target].astype(int)
    X_test = test_df[feature_columns].copy()
    y_test = test_df[target].astype(int)

    # LightGBM does not allow special JSON characters in feature names.
    # Sanitize feature names consistently for both train and test sets.
    safe_feature_names = (
        pd.Index(X_train.columns)
        .astype(str)
        .str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)
    )

    # Ensure sanitized feature names remain unique.
    if safe_feature_names.duplicated().any():
        counts = {}
        unique_names = []
        for name in safe_feature_names:
            count = counts.get(name, 0)
            unique_names.append(name if count == 0 else f"{name}_{count}")
            counts[name] = count + 1
        safe_feature_names = unique_names

    X_train.columns = safe_feature_names
    X_test.columns = safe_feature_names

    non_numeric = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(
            "Processed model inputs must be numeric. Non-numeric columns found: "
            f"{non_numeric}"
        )

    if sorted(y_train.unique().tolist()) != [0, 1]:
        raise ValueError(f"Training target must be binary 0/1; got {sorted(y_train.unique())}")
    if sorted(y_test.unique().tolist()) != [0, 1]:
        raise ValueError(f"Test target must be binary 0/1; got {sorted(y_test.unique())}")

    return X_train, y_train, X_test, y_test


def _positive_class_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{type(model).__name__} does not implement predict_proba().")
    probabilities = np.asarray(model.predict_proba(X))
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError(
            f"Expected binary predict_proba output with shape (n, 2), got {probabilities.shape}."
        )
    return probabilities[:, 1]


def _fit_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    balance_training: bool,
) -> float:
    sample_weight = compute_sample_weight(class_weight="balanced", y=y) if balance_training else None
    start = time.perf_counter()
    if sample_weight is None:
        model.fit(X, y)
    else:
        model.fit(X, y, sample_weight=sample_weight)
    return time.perf_counter() - start


def _run_one_model(
    *,
    dataset_key: str,
    dataset_name: str,
    model_key: str,
    model_config: dict[str, Any],
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    experiment_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    random_state = int(experiment_config["random_state"])
    threshold = float(experiment_config["decision_threshold"])
    balance_training = bool(experiment_config["balance_training"])
    cv_config = experiment_config["cv"]
    model_name = model_config.get("display_name", model_key)

    splitter = StratifiedKFold(
        n_splits=int(cv_config["n_splits"]),
        shuffle=bool(cv_config["shuffle"]),
        random_state=random_state if bool(cv_config["shuffle"]) else None,
    )

    fold_rows: list[dict[str, Any]] = []
    shap_rows: list[dict[str, Any]] = []
    for fold_index, (train_indices, validation_indices) in enumerate(
        splitter.split(X_dev, y_dev), start=1
    ):
        X_fold_train = X_dev.iloc[train_indices]
        y_fold_train = y_dev.iloc[train_indices]
        X_fold_validation = X_dev.iloc[validation_indices]
        y_fold_validation = y_dev.iloc[validation_indices]

        model = build_model(model_key, model_config, random_state=random_state)
        fit_seconds = _fit_model(
            model,
            X_fold_train,
            y_fold_train,
            balance_training=balance_training,
        )
        validation_score = _positive_class_probability(model, X_fold_validation)
        metrics = compute_binary_metrics(
            y_fold_validation,
            validation_score,
            threshold=threshold,
        )
        shap_importance = compute_fold_shap_importance(model, X_fold_validation)
        for feature, importance in shap_importance.items():
            shap_rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset_name": dataset_name,
                    "model_key": model_key,
                    "model_name": model_name,
                    "fold": fold_index,
                    "feature": feature,
                    "mean_abs_shap": float(importance),
                }
            )

        fold_rows.append(
            {
                "dataset_key": dataset_key,
                "dataset_name": dataset_name,
                "model_key": model_key,
                "model_name": model_name,
                "fold": fold_index,
                "n_train": len(train_indices),
                "n_validation": len(validation_indices),
                "train_positive_rate": float(y_fold_train.mean()),
                "validation_positive_rate": float(y_fold_validation.mean()),
                "fit_seconds": fit_seconds,
                **metrics,
            }
        )
        metric_text = ", ".join(f"{name}={metrics[name]:.4f}" for name in METRIC_NAMES)
        print(
            f"    Fold {fold_index}/{cv_config['n_splits']} | "
            f"{metric_text} | fit={fit_seconds:.2f}s"
        )

    final_model = build_model(model_key, model_config, random_state=random_state)
    final_fit_seconds = _fit_model(
        final_model,
        X_dev,
        y_dev,
        balance_training=balance_training,
    )
    test_score = _positive_class_probability(final_model, X_test)
    test_metrics = compute_binary_metrics(y_test, test_score, threshold=threshold)
    test_row = {
        "dataset_key": dataset_key,
        "dataset_name": dataset_name,
        "model_key": model_key,
        "model_name": model_name,
        "n_development": len(X_dev),
        "n_test": len(X_test),
        "development_positive_rate": float(y_dev.mean()),
        "test_positive_rate": float(y_test.mean()),
        "fit_seconds": final_fit_seconds,
        **test_metrics,
    }
    metric_text = ", ".join(f"{name}={test_metrics[name]:.4f}" for name in METRIC_NAMES)
    print(f"    Hold-out test | {metric_text} | fit={final_fit_seconds:.2f}s")
    return fold_rows, shap_rows, test_row


def _merge_test_metrics(ranked_summary: pd.DataFrame, test_metrics: pd.DataFrame) -> pd.DataFrame:
    test_columns = ["dataset_key", "model_key", *METRIC_NAMES]
    renamed = test_metrics[test_columns].rename(
        columns={metric: f"test_{metric}" for metric in METRIC_NAMES}
    )
    return ranked_summary.merge(renamed, on=["dataset_key", "model_key"], how="left")


def _write_metadata(config_path: Path, output_dir: Path) -> None:
    config_bytes = config_path.read_bytes()
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    with (output_dir / "experiment_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)


def run_experiments(
    config_path: str | Path,
    selected_datasets: set[str] | None = None,
    selected_models: set[str] | None = None,
) -> dict[str, Path]:
    config, project_root = load_experiment_config(config_path)
    experiment = config["experiment"]
    datasets = config["datasets"]
    models = config["models"]

    if selected_datasets:
        unknown = selected_datasets.difference(datasets)
        if unknown:
            raise ValueError(f"Unknown dataset keys: {sorted(unknown)}")
        datasets = {key: value for key, value in datasets.items() if key in selected_datasets}
    if selected_models:
        unknown = selected_models.difference(models)
        if unknown:
            raise ValueError(f"Unknown model keys: {sorted(unknown)}")
        models = {key: value for key, value in models.items() if key in selected_models}

    output_dir = project_root / experiment["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    all_fold_rows: list[dict[str, Any]] = []
    all_shap_rows: list[dict[str, Any]] = []
    all_test_rows: list[dict[str, Any]] = []

    for dataset_key, dataset_config in datasets.items():
        print(f"\n=== {dataset_config['name']} ({dataset_key}) ===")
        X_dev, y_dev, X_test, y_test = _load_processed_split(project_root, dataset_config)
        print(
            f"Development: {X_dev.shape} | Test: {X_test.shape} | "
            f"positive rate={y_dev.mean():.4f}/{y_test.mean():.4f}"
        )

        for model_key, model_config in models.items():
            print(f"  -> {model_config.get('display_name', model_key)}")
            fold_rows, shap_rows, test_row = _run_one_model(
                dataset_key=dataset_key,
                dataset_name=dataset_config["name"],
                model_key=model_key,
                model_config=model_config,
                X_dev=X_dev,
                y_dev=y_dev,
                X_test=X_test,
                y_test=y_test,
                experiment_config=experiment,
            )
            all_fold_rows.extend(fold_rows)
            all_shap_rows.extend(shap_rows)
            all_test_rows.append(test_row)

    fold_metrics = pd.DataFrame(all_fold_rows)
    test_metrics = pd.DataFrame(all_test_rows)
    shap_fold_importance = pd.DataFrame(all_shap_rows)
    shap_stability = summarize_shap_stability(
        shap_fold_importance,
        top_k=int(experiment.get("shap", {}).get("top_k", 10)),
    )
    cv_summary = summarize_fold_metrics(fold_metrics)
    ranked_summary = add_dataset_ranks(cv_summary, primary_metric=experiment["primary_metric"])
    model_comparison = _merge_test_metrics(ranked_summary, test_metrics)
    overall_comparison = build_overall_comparison(ranked_summary)

    paths = {
        "fold_metrics": output_dir / "fold_metrics.csv",
        "cv_summary": output_dir / "cv_summary.csv",
        "test_metrics": output_dir / "test_metrics.csv",
        "model_comparison": output_dir / "model_comparison.csv",
        "overall_model_comparison": output_dir / "overall_model_comparison.csv",
        "shap_fold_importance": output_dir / "shap_fold_importance.csv",
        "shap_stability": output_dir / "shap_stability.csv",
        "metadata": output_dir / "experiment_metadata.json",
    }
    fold_metrics.to_csv(paths["fold_metrics"], index=False)
    cv_summary.to_csv(paths["cv_summary"], index=False)
    test_metrics.to_csv(paths["test_metrics"], index=False)
    model_comparison.to_csv(paths["model_comparison"], index=False)
    overall_comparison.to_csv(paths["overall_model_comparison"], index=False)
    shap_fold_importance.to_csv(paths["shap_fold_importance"], index=False)
    shap_stability.to_csv(paths["shap_stability"], index=False)
    _write_metadata(Path(config_path).expanduser().resolve(), output_dir)

    print("\n=== Cross-dataset comparison ===")
    columns = ["model_name", "mean_dataset_rank", "pr_auc_mean", "roc_auc_mean", "recall_mean", "f1_mean"]
    print(overall_comparison[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nResults saved to: {output_dir}")
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AdaBoost, XGBoost, and LightGBM healthcare-risk experiments."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to experiment YAML config.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset keys to run (e.g. dataset1 dataset2).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model keys to run (adaboost xgboost lightgbm).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_experiments(
        config_path=args.config,
        selected_datasets=set(args.datasets) if args.datasets else None,
        selected_models=set(args.models) if args.models else None,
    )


if __name__ == "__main__":
    main()
