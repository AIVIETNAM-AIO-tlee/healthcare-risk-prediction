from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import compute_binary_metrics, summarize_fold_metrics


def test_compute_binary_metrics_perfect_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.01, 0.10, 0.90, 0.99])

    metrics = compute_binary_metrics(y_true, y_score, threshold=0.5)

    assert metrics == {
        "roc_auc": pytest.approx(1.0),
        "pr_auc": pytest.approx(1.0),
        "recall": pytest.approx(1.0),
        "f1": pytest.approx(1.0),
    }


def test_summarize_fold_metrics_reports_variability() -> None:
    fold_metrics = pd.DataFrame(
        [
            {
                "dataset_key": "dataset1",
                "dataset_name": "Example",
                "model_key": "xgboost",
                "model_name": "XGBoost",
                "roc_auc": 0.80,
                "pr_auc": 0.60,
                "recall": 0.70,
                "f1": 0.65,
            },
            {
                "dataset_key": "dataset1",
                "dataset_name": "Example",
                "model_key": "xgboost",
                "model_name": "XGBoost",
                "roc_auc": 0.90,
                "pr_auc": 0.70,
                "recall": 0.80,
                "f1": 0.75,
            },
        ]
    )

    summary = summarize_fold_metrics(fold_metrics)

    assert summary.loc[0, "roc_auc_mean"] == pytest.approx(0.85)
    assert summary.loc[0, "roc_auc_range"] == pytest.approx(0.10)
    assert summary.loc[0, "pr_auc_mean"] == pytest.approx(0.65)
