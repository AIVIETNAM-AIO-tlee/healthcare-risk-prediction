---
title: "Tree-based Models Healthcare Risk Prediction: CV and SHAP Stability Study"
description: "Comparative study of Decision Tree, Random Forest, XGBoost across 6 healthcare datasets: repeated nested CV performance comparison (RQ1), performance stability (RQ2), SHAP ranking stability (RQ3). Mode: --tdd."
status: pending
priority: P2
branch: "main"
tags: [research, ml, shap, cross-validation]
blockedBy: []
blocks: []
created: "2026-08-23T12:43:33.975Z"
createdBy: "ck:plan"
source: skill
---

# Tree-based Models Healthcare Risk Prediction: CV and SHAP Stability Study

## Overview

Build a reproducible Python research package (`tree_risk_stability`) that answers three research questions:

- **RQ1**: How do Decision Tree, Random Forest, and XGBoost compare in healthcare risk prediction across multiple datasets?
- **RQ2**: How stable is predictive performance across CV folds?
- **RQ3**: How stable are SHAP-based feature importance rankings across CV folds?

Datasets (6): synthetic repo `Santy3298/healthcare-ai-risk-dataset` (n=50, multiclass Low/Med/High), UCI Heart Disease (303), Breast Cancer Wisconsin (569), Pima Diabetes (768), Framingham (~4.2k), Stroke Prediction (~5.1k).

Protocol: Repeated Stratified 5-Fold × 10 repeats, nested CV (inner 3-fold grid search), preprocessing inside sklearn Pipelines (zero leakage), SHAP TreeExplainer per fold, Kendall's W + pairwise Spearman + Jaccard@5/@10 + mean|SHAP| variance for RQ3, Friedman/Nemenyi + Wilcoxon-Holm inference, critical difference diagrams.

Context: brainstorm report at `plans/260823-1936-tree-risk-stability-brainstorm/tree-risk-stability-brainstorm-report.md` (approved design).

## TDD Mode Note

Every code-bearing phase follows red-green-refactor: failing tests written BEFORE implementation, committed together. Metric implementations get property-based and known-analytic-value tests so statistical code is verifiable against textbook examples.

## Phases

| Phase | Name | Status |
|-------|------|--------|
| 1 | [Research](./phase-01-research.md) | Pending |
| 2 | [Data Layer](./phase-02-data-layer.md) | Pending |
| 3 | [Model Pipeline](./phase-03-model-pipeline.md) | Pending |
| 4 | [Explainability](./phase-04-explainability.md) | Pending |
| 5 | [Statistical Evaluation](./phase-05-statistical-evaluation.md) | Pending |
| 6 | [Analysis and Paper](./phase-06-analysis-and-paper.md) | Pending |

## Dependencies

- Linear chain: 1 → 2 → 3 → 4 → 5 → 6 (each phase consumes prior outputs; no parallel phases required)
- No cross-plan dependencies (fresh repo; only related artifact is the brainstorm report, which is input context not a blocking plan)
