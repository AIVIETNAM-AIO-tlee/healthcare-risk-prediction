from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


def plot_roc_curves(
    curves: dict[str, dict[str, Any]],
    output_path: Path,
    title: str,
) -> Path:
    """Plot model ROC curves from binary labels and positive-class scores."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, values in curves.items():
        y_true = np.asarray(values["y_true"], dtype=int)
        y_score = np.asarray(values["y_score"], dtype=float)
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_score)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        ax.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2,
            label=f"{model_name} (AUC = {roc_auc:.4f})",
        )

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Random classifier")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_dataset_roc_curves(
    dataset_curves: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
) -> dict[str, Path]:
    """Save one comparison plot per dataset and one panel plot for all datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for dataset_key, curves in dataset_curves.items():
        path = output_dir / f"roc_curves_{dataset_key}.png"
        plot_roc_curves(
            curves,
            path,
            title=f"ROC Curves - {dataset_key} Hold-out Test Set",
        )
        paths[dataset_key] = path

    if dataset_curves:
        fig, axes = plt.subplots(
            1,
            len(dataset_curves),
            figsize=(7 * len(dataset_curves), 5.5),
            squeeze=False,
        )
        for axis, (dataset_key, curves) in zip(axes[0], dataset_curves.items()):
            for model_name, values in curves.items():
                y_true = np.asarray(values["y_true"], dtype=int)
                y_score = np.asarray(values["y_score"], dtype=float)
                false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_score)
                roc_auc = auc(false_positive_rate, true_positive_rate)
                axis.plot(
                    false_positive_rate,
                    true_positive_rate,
                    linewidth=2,
                    label=f"{model_name} (AUC = {roc_auc:.4f})",
                )
            axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
            axis.set_title(dataset_key)
            axis.set_xlabel("False Positive Rate")
            axis.set_ylabel("True Positive Rate")
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.02)
            axis.grid(alpha=0.25)
            axis.legend(loc="lower right")

        fig.suptitle("ROC Curves Across Healthcare Datasets", fontsize=14)
        fig.tight_layout()
        combined_path = output_dir / "roc_curves_all_datasets.png"
        fig.savefig(combined_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths["all_datasets"] = combined_path

    return paths
