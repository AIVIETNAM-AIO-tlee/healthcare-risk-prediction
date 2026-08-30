from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from src.explainability.shap_analysis import (
    compute_fold_shap_importance,
    compute_pairwise_rank_stability,
    summarize_consensus_ranking,
    summarize_rank_stability,
)
from src.experiment_config import load_experiment_config
from src.experiments.run_models import _load_processed_split
from src.models.factory import build_model


def _fit_for_shap(model: Any, X: pd.DataFrame, y: pd.Series, balance_training: bool) -> None:
    weights = compute_sample_weight(class_weight="balanced", y=y) if balance_training else None
    if weights is None:
        model.fit(X, y)
    else:
        model.fit(X, y, sample_weight=weights)


def run_shap_experiments(
    config_path: str | Path,
    selected_datasets: set[str] | None = None,
    selected_models: set[str] | None = None,
) -> dict[str, Path]:
    """Run fold-wise SHAP explainability and ranking-stability analysis for RQ3."""
    config, project_root = load_experiment_config(config_path)
    experiment = config["experiment"]
    datasets = config["datasets"]
    models = config["models"]
    shap_config = config.get("shap")
    if not shap_config:
        raise ValueError("Missing top-level 'shap' configuration in config.yaml.")

    if selected_datasets:
        unknown = selected_datasets.difference(datasets)
        if unknown:
            raise ValueError(f"Unknown dataset keys: {sorted(unknown)}")
        datasets = {k: v for k, v in datasets.items() if k in selected_datasets}
    if selected_models:
        unknown = selected_models.difference(models)
        if unknown:
            raise ValueError(f"Unknown model keys: {sorted(unknown)}")
        models = {k: v for k, v in models.items() if k in selected_models}

    random_state = int(experiment["random_state"])
    cv_config = experiment["cv"]
    balance_training = bool(experiment["balance_training"])
    max_explain_samples = int(shap_config["max_explain_samples"])
    background_samples = int(shap_config["background_samples"])
    top_k = int(shap_config["top_k"])

    output_dir = project_root / shap_config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    all_importance_rows: list[pd.DataFrame] = []
    execution_rows: list[dict[str, Any]] = []

    for dataset_key, dataset_config in datasets.items():
        print(f"\n=== SHAP: {dataset_config['name']} ({dataset_key}) ===")
        X_dev, y_dev, _, _ = _load_processed_split(project_root, dataset_config)
        splitter = StratifiedKFold(
            n_splits=int(cv_config["n_splits"]),
            shuffle=bool(cv_config["shuffle"]),
            random_state=random_state if bool(cv_config["shuffle"]) else None,
        )
        splits = list(splitter.split(X_dev, y_dev))

        for model_key, model_config in models.items():
            model_name = model_config.get("display_name", model_key)
            print(f"  -> {model_name}")
            for fold_index, (train_idx, validation_idx) in enumerate(splits, start=1):
                X_train = X_dev.iloc[train_idx]
                y_train = y_dev.iloc[train_idx]
                X_validation = X_dev.iloc[validation_idx]

                model = build_model(model_key, model_config, random_state=random_state)
                _fit_for_shap(model, X_train, y_train, balance_training)
                importance, meta = compute_fold_shap_importance(
                    model=model,
                    model_key=model_key,
                    X_train=X_train,
                    X_validation=X_validation,
                    max_explain_samples=max_explain_samples,
                    background_samples=background_samples,
                    random_state=random_state + fold_index,
                )
                importance.insert(0, "fold", fold_index)
                importance.insert(0, "model_name", model_name)
                importance.insert(0, "model_key", model_key)
                importance.insert(0, "dataset_name", dataset_config["name"])
                importance.insert(0, "dataset_key", dataset_key)
                all_importance_rows.append(importance)

                execution_rows.append(
                    {
                        "dataset_key": dataset_key,
                        "dataset_name": dataset_config["name"],
                        "model_key": model_key,
                        "model_name": model_name,
                        "fold": fold_index,
                        **meta,
                    }
                )
                print(
                    f"    Fold {fold_index}/{cv_config['n_splits']} | "
                    f"{meta['explainer']} | explained={meta['n_explained']} | "
                    f"top feature={importance.iloc[0]['feature']}"
                )

    feature_importance = pd.concat(all_importance_rows, ignore_index=True)
    pairwise = compute_pairwise_rank_stability(feature_importance, top_k=top_k)
    stability_summary = summarize_rank_stability(pairwise)
    consensus = summarize_consensus_ranking(feature_importance, top_k=top_k)
    execution = pd.DataFrame(execution_rows)

    paths = {
        "feature_importance": output_dir / "shap_feature_importance.csv",
        "pairwise_stability": output_dir / "shap_fold_stability.csv",
        "stability_summary": output_dir / "shap_stability_summary.csv",
        "consensus_ranking": output_dir / "shap_consensus_ranking.csv",
        "execution": output_dir / "shap_execution_summary.csv",
        "metadata": output_dir / "shap_metadata.json",
    }
    feature_importance.to_csv(paths["feature_importance"], index=False)
    pairwise.to_csv(paths["pairwise_stability"], index=False)
    stability_summary.to_csv(paths["stability_summary"], index=False)
    consensus.to_csv(paths["consensus_ranking"], index=False)
    execution.to_csv(paths["execution"], index=False)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": random_state,
        "cv_folds": int(cv_config["n_splits"]),
        "max_explain_samples_per_fold": max_explain_samples,
        "background_samples_for_adaboost": background_samples,
        "top_k": top_k,
        "note": (
            "XGBoost/LightGBM use TreeExplainer; AdaBoost uses permutation SHAP "
            "because sklearn AdaBoost is not supported by TreeExplainer in SHAP 0.50."
        ),
    }
    with paths["metadata"].open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    print("\n=== SHAP ranking stability summary ===")
    display_cols = [
        "dataset_key", "model_name", "spearman_mean", "top_k_jaccard_mean"
    ]
    print(stability_summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSHAP results saved to: {output_dir}")
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fold-wise SHAP explainability and ranking-stability analysis."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_shap_experiments(
        config_path=args.config,
        selected_datasets=set(args.datasets) if args.datasets else None,
        selected_models=set(args.models) if args.models else None,
    )


if __name__ == "__main__":
    main()
