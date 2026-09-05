"""Evaluation metrics tests (scope: 'Test evaluation metrics ROC-AUC, PR-AUC,
Recall, F1' + 'Validate model results and output formats').

Covers the metric math on hand-computable cases, range/error contracts, and
the fold-summary / ranking / cross-dataset aggregation formats consumed by the
results CSVs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.metrics import (
    METRIC_NAMES,
    add_dataset_ranks,
    build_overall_comparison,
    compute_binary_metrics,
    summarize_fold_metrics,
)


def _hand_case():
    y_true = pd.Series([0, 0, 1, 1, 0, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8, 0.65, 0.9])
    # at threshold 0.5 -> y_pred = [0, 0, 0, 1, 1, 1]
    return y_true, y_score


def test_metrics_match_hand_computed_values():
    y_true, y_score = _hand_case()

    metrics = compute_binary_metrics(y_true, y_score)

    # ROC-AUC from positive ranks (2, 5, 6) over 3/3 class split: (13 - 6) / 9
    assert metrics["roc_auc"] == pytest.approx(7 / 9)
    # recall = TP 2 of 3 positives; precision 2/3 -> F1 = 2/3
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    # PR curve trapezoid: (0,1)->(1/3,1)->(2/3,1)->(2/3,1/2)->(1,3/5)->(1,1/2)
    assert metrics["pr_auc"] == pytest.approx(0.85)


def test_perfect_classifier_scores_one_everywhere():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = compute_binary_metrics(y_true, y_score)

    assert all(metrics[name] == pytest.approx(1.0) for name in METRIC_NAMES)


def test_all_metrics_are_within_unit_range():
    y_true, y_score = _hand_case()

    metrics = compute_binary_metrics(y_true, y_score)

    assert all(0.0 <= metrics[name] <= 1.0 for name in METRIC_NAMES)


def test_threshold_changes_recall_and_f1_but_not_auc_metrics():
    y_true, y_score = _hand_case()
    base = compute_binary_metrics(y_true, y_score)

    lowered = compute_binary_metrics(y_true, y_score, threshold=0.3)

    assert lowered["roc_auc"] == base["roc_auc"]
    assert lowered["pr_auc"] == base["pr_auc"]
    assert lowered["recall"] == 1.0  # every positive now predicted positive
    assert lowered["f1"] > base["f1"]


def test_constant_scores_yield_chance_roc():
    y_true = np.array([0, 1, 0, 1])

    metrics = compute_binary_metrics(y_true, np.full(4, 0.5))

    assert metrics["roc_auc"] == pytest.approx(0.5)
    assert metrics["recall"] == 1.0  # all rows cross the 0.5 threshold


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same number of rows"):
        compute_binary_metrics([0, 1], np.array([0.5]))


def test_single_class_target_raises():
    with pytest.raises(ValueError, match="both classes"):
        compute_binary_metrics([1, 1, 1], np.array([0.2, 0.8, 0.5]))


def test_two_dimensional_input_raises():
    with pytest.raises(ValueError, match="one-dimensional"):
        compute_binary_metrics([[0], [1]], np.array([[0.5], [0.6]]))


# ---------------------------------------------------------------------------
# Result-format aggregations
# ---------------------------------------------------------------------------
def _fold_frame() -> pd.DataFrame:
    rows = []
    scores = {
        "dt": [(0.8, 0.7), (0.6, 0.5)],  # (roc_auc, pr_auc) per fold
        "rf": [(0.7, 0.8), (0.9, 0.9)],
    }
    for model, fold_scores in scores.items():
        for fold, (roc, pr) in enumerate(fold_scores, start=1):
            rows.append(
                {
                    "dataset_key": "ds1",
                    "dataset_name": "Dataset 1",
                    "model_key": model,
                    "model_name": model.upper(),
                    "fold": fold,
                    "roc_auc": roc,
                    "pr_auc": pr,
                    "recall": 0.5,
                    "f1": 0.5,
                }
            )
    return pd.DataFrame(rows)


def test_summarize_fold_metrics_produces_stability_columns():
    summary = summarize_fold_metrics(_fold_frame())

    dt = summary[summary["model_key"] == "dt"].iloc[0]
    assert dt["roc_auc_mean"] == pytest.approx(0.7)
    assert dt["roc_auc_std"] == pytest.approx(pd.Series([0.8, 0.6]).std())
    assert dt["roc_auc_min"] == 0.6 and dt["roc_auc_max"] == 0.8
    assert dt["roc_auc_range"] == pytest.approx(0.2)
    assert "pr_auc_range" in summary.columns and "f1_range" in summary.columns


def test_summarize_missing_required_column_raises():
    frame = _fold_frame().drop(columns=["f1"])

    with pytest.raises(ValueError, match="f1"):
        summarize_fold_metrics(frame)


def test_add_dataset_ranks_orders_by_primary_metric():
    summary = add_dataset_ranks(summarize_fold_metrics(_fold_frame()), "pr_auc")

    rf = summary[summary["model_key"] == "rf"].iloc[0]
    dt = summary[summary["model_key"] == "dt"].iloc[0]
    assert rf["cv_rank"] == 1 and dt["cv_rank"] == 2


def test_add_dataset_ranks_rejects_unknown_primary_metric():
    with pytest.raises(ValueError, match="Unsupported primary metric"):
        add_dataset_ranks(summarize_fold_metrics(_fold_frame()), "accuracy")


def test_overall_comparison_averages_rank_across_datasets():
    summary = summarize_fold_metrics(_fold_frame())
    # second dataset with the two models' identities swapped, so each model
    # is rank 1 in one dataset and rank 2 in the other (key AND display name
    # must swap together to keep the aggregation groups coherent)
    swapped = summarize_fold_metrics(
        _fold_frame()
        .assign(
            dataset_key="ds2",
            dataset_name="Dataset 2",
            model_key=lambda frame: frame["model_key"].map({"dt": "rf", "rf": "dt"}),
            model_name=lambda frame: frame["model_name"].map({"DT": "RF", "RF": "DT"}),
        )
    )
    both = add_dataset_ranks(pd.concat([summary, swapped], ignore_index=True), "pr_auc")

    overall = build_overall_comparison(both)

    assert set(overall["model_key"]) == {"dt", "rf"}
    assert "mean_dataset_rank" in overall.columns
    # rank 2 in ds1 + rank 1 in ds2 -> mean 1.5 for dt
    dt_rank = overall.loc[overall["model_key"] == "dt", "mean_dataset_rank"].iloc[0]
    assert dt_rank == pytest.approx(1.5)
