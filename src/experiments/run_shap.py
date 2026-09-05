from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.utils.class_weight import compute_sample_weight

from src.explainability.shap_analysis import (
    compute_fold_shap_importance,
    summarize_consensus_ranking,
)
from src.explainability.shap_stability import (
    build_fold_correlation_matrix,
    build_rank_matrix,
    compute_pairwise_rank_stability,
    summarize_rank_stability,
)
from src.experiment_config import load_experiment_config
from src.experiments.run_models import _load_kfold_splits, _load_processed_split
from src.models.factory import build_model


def _fit_for_shap(model: Any, X: pd.DataFrame, y: pd.Series, balance_training: bool) -> None:
    weights = compute_sample_weight(class_weight="balanced", y=y) if balance_training else None
    if weights is None:
        model.fit(X, y)
    else:
        model.fit(X, y, sample_weight=weights)


def _render_matrix_heatmap(
    *,
    matrix: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
    output_path: Path,
    annotation_format: str,
) -> Path:
    """Render a simple annotated heatmap using only matplotlib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = matrix.to_numpy(dtype=float)

    fig_width = max(8.0, 0.65 * len(matrix.columns) + 2.5)
    fig_height = max(5.0, 0.55 * len(matrix.index) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(values, aspect="auto")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if values.size:
        threshold = np.nanmedian(values)
    else:
        threshold = 0.0
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isnan(value):
                label = "NA"
            else:
                label = format(value, annotation_format)
            text_color = "white" if not np.isnan(value) and value >= threshold else "black"
            ax.text(col, row, label, ha="center", va="center", fontsize=9, color=text_color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_global_shap_plots(
    *,
    shap_values: np.ndarray,
    feature_values: pd.DataFrame,
    dataset_key: str,
    model_key: str,
    model_name: str,
    dataset_name: str,
    top_k: int,
    plot_dir: Path,
) -> dict[str, Path]:
    """Save global SHAP explainability plots for one dataset/model pair."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = f"{dataset_key}_{model_key}"
    display_k = min(top_k, feature_values.shape[1])

    beeswarm_path = plot_dir / f"{safe_stem}_beeswarm.png"
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        feature_values,
        max_display=display_k,
        show=False,
        plot_type="dot",
    )
    plt.title(f"SHAP Beeswarm — {model_name} / {dataset_name}")
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=200, bbox_inches="tight")
    plt.close()

    bar_path = plot_dir / f"{safe_stem}_bar.png"
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        feature_values,
        max_display=display_k,
        show=False,
        plot_type="bar",
    )
    plt.title(f"Mean |SHAP| — {model_name} / {dataset_name}")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200, bbox_inches="tight")
    plt.close()

    violin_path = plot_dir / f"{safe_stem}_violin.png"
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        feature_values,
        max_display=display_k,
        show=False,
        plot_type="violin",
    )
    plt.title(f"SHAP Violin Summary — {model_name} / {dataset_name}")
    plt.tight_layout()
    plt.savefig(violin_path, dpi=200, bbox_inches="tight")
    plt.close()

    return {"beeswarm": beeswarm_path, "bar": bar_path, "violin": violin_path}


def _save_stability_plots(
    *,
    feature_importance: pd.DataFrame,
    dataset_key: str,
    dataset_name: str,
    model_key: str,
    model_name: str,
    top_k: int,
    plot_dir: Path,
) -> dict[str, Path]:
    """Save SHAP ranking heatmap and fold-correlation heatmap."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = f"{dataset_key}_{model_key}"

    rank_matrix = build_rank_matrix(
        feature_importance,
        dataset_key=dataset_key,
        model_key=model_key,
        max_features=max(top_k, min(15, feature_importance[feature_importance['dataset_key'].eq(dataset_key) & feature_importance['model_key'].eq(model_key)]['feature'].nunique())),
    )
    rank_heatmap_path = plot_dir / f"{safe_stem}_ranking_heatmap.png"
    _render_matrix_heatmap(
        matrix=rank_matrix,
        title=f"Feature-Ranking Heatmap — {model_name} / {dataset_name}",
        xlabel="Feature",
        ylabel="Fold",
        colorbar_label="Rank (1 = most important)",
        output_path=rank_heatmap_path,
        annotation_format=".0f",
    )

    corr_matrix = build_fold_correlation_matrix(
        feature_importance,
        dataset_key=dataset_key,
        model_key=model_key,
        method="spearman",
    )
    corr_heatmap_path = plot_dir / f"{safe_stem}_fold_correlation_heatmap.png"
    _render_matrix_heatmap(
        matrix=corr_matrix,
        title=f"Fold Correlation Heatmap (Spearman) — {model_name} / {dataset_name}",
        xlabel="Validation fold",
        ylabel="Validation fold",
        colorbar_label="Spearman rank correlation",
        output_path=corr_heatmap_path,
        annotation_format=".2f",
    )

    return {
        "ranking_heatmap": rank_heatmap_path,
        "fold_correlation_heatmap": corr_heatmap_path,
    }


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
    plot_paths: dict[str, Path] = {}
    plot_dir = output_dir / "plots"

    for dataset_key, dataset_config in datasets.items():
        print(f"\n=== SHAP: {dataset_config['name']} ({dataset_key}) ===")
        X_dev, y_dev, _, _ = _load_processed_split(project_root, dataset_config)
        splits = _load_kfold_splits(
            project_root=project_root,
            dataset_config=dataset_config,
            n_rows=len(X_dev),
            expected_n_splits=int(cv_config["n_splits"]),
        )

        for model_key, model_config in models.items():
            model_name = model_config.get("display_name", model_key)
            print(f"  -> {model_name}")
            model_shap_values: list[np.ndarray] = []
            model_feature_values: list[pd.DataFrame] = []
            for fold_index, (train_idx, validation_idx) in enumerate(splits, start=1):
                X_train = X_dev.iloc[train_idx]
                y_train = y_dev.iloc[train_idx]
                X_validation = X_dev.iloc[validation_idx]

                model = build_model(model_key, model_config, random_state=random_state)
                _fit_for_shap(model, X_train, y_train, balance_training)
                importance, meta, fold_shap_values, fold_feature_values = compute_fold_shap_importance(
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
                model_shap_values.append(fold_shap_values)
                model_feature_values.append(fold_feature_values)

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

            combined_shap = np.vstack(model_shap_values)
            combined_features = pd.concat(model_feature_values, ignore_index=True)
            saved = _save_global_shap_plots(
                shap_values=combined_shap,
                feature_values=combined_features,
                dataset_key=dataset_key,
                dataset_name=dataset_config["name"],
                model_key=model_key,
                model_name=model_name,
                top_k=top_k,
                plot_dir=plot_dir,
            )
            for plot_name, plot_path in saved.items():
                plot_paths[f"{dataset_key}_{model_key}_{plot_name}"] = plot_path

    feature_importance = pd.concat(all_importance_rows, ignore_index=True)
    pairwise = compute_pairwise_rank_stability(feature_importance, top_k=top_k)
    stability_summary = summarize_rank_stability(pairwise)
    consensus = summarize_consensus_ranking(feature_importance, top_k=top_k)
    execution = pd.DataFrame(execution_rows)

    for keys, group in feature_importance.groupby(["dataset_key", "dataset_name", "model_key", "model_name"], sort=False):
        dataset_key, dataset_name, model_key, model_name = keys
        saved = _save_stability_plots(
            feature_importance=group,
            dataset_key=dataset_key,
            dataset_name=dataset_name,
            model_key=model_key,
            model_name=model_name,
            top_k=top_k,
            plot_dir=plot_dir,
        )
        for plot_name, plot_path in saved.items():
            plot_paths[f"{dataset_key}_{model_key}_{plot_name}"] = plot_path

    paths = {
        "feature_importance": output_dir / "shap_feature_importance.csv",
        "pairwise_stability": output_dir / "shap_fold_stability.csv",
        "stability_summary": output_dir / "shap_stability_summary.csv",
        "consensus_ranking": output_dir / "shap_consensus_ranking.csv",
        "execution": output_dir / "shap_execution_summary.csv",
        "metadata": output_dir / "shap_metadata.json",
        **plot_paths,
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
        "plots": {key: str(path.relative_to(project_root)) for key, path in plot_paths.items()},
        "note": (
            "XGBoost/LightGBM use TreeExplainer; AdaBoost uses permutation SHAP "
            "because sklearn AdaBoost is not supported by TreeExplainer in SHAP 0.50."
        ),
    }
    with paths["metadata"].open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    print("\n=== SHAP ranking stability summary ===")
    display_cols = [
        "dataset_key",
        "model_name",
        "kendall_tau_mean",
        "spearman_mean",
        "top_k_jaccard_mean",
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
