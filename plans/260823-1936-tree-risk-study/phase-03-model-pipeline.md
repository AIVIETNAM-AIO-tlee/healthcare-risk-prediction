---
phase: 3
title: "Model Pipeline"
status: pending
priority: P1
effort: "3d"
dependencies: ["phase-02-data-layer.md"]
---

# Phase 3: Model Pipeline

## Overview

TDD-build the experiment engine: model factories with fixed grids, leakage-free sklearn Pipelines, and the repeated nested-CV runner that produces per-fold performance records (RQ1/RQ2 raw data) plus per-fold fitted models for Phase 4.

## Requirements

- Functional:
  - `make_model(name, params) -> sklearn Pipeline` for {"decision_tree", "random_forest", "xgboost"}; preprocessing steps (impute + encode) inside the Pipeline.
  - `run_nested_cv(bundle, config) -> list[FoldRecord]`; `FoldRecord` = dataclass: dataset, model, repeat, fold, metrics dict {roc_auc, weighted_f1, balanced_accuracy, mcc}, best_params, fitted estimator ref or path.
  - Repeated Stratified 5-Fold × 10 repeats; inner GridSearchCV 3-fold on train folds only; seeds deterministic from base seed.
- Non-functional: resumable — results appended to JSONL per (dataset, model); a killed run restarts without duplicating completed work.

## Architecture

```
src/tree_risk_stability/models/
├── factories.py     # model + preprocessing pipeline builders, GRIDS dict
├── nested_cv.py     # run_nested_cv, FoldRecord, seed derivation
└── runner.py        # CLI entrypoint: --config <yaml> [--models dt,rf,xgb]
results/raw/
├── <slug>__<model>.jsonl   # one FoldRecord per line
```

Metric computation via a single `compute_metrics(y_true, y_pred, y_proba, task_type)` in `evaluation/metrics.py` (built here, reused everywhere). Multiclass ROC-AUC = macro OVR.

## Related Code Files

- Create: `src/tree_risk_stability/models/{factories,nested_cv,runner}.py`
- Create: `src/tree_risk_stability/evaluation/metrics.py`
- Create: `tests/models/test_factories.py`, `tests/models/test_nested_cv.py`, `tests/evaluation/test_metrics.py`

## Implementation Steps

1. **RED**: `test_metrics.py` — known-analytic cases: perfect predictions → AUC=1.0, MCC undefined-safe (returns 0 on degenerate), balanced accuracy on hand-built 4-row case, macro-OVR multiclass case verified against manual one-vs-rest computation; property test: metric values invariant to label permutation of classes in multiclass mode.
2. **GREEN**: Implement `metrics.py`.
3. **RED**: `test_factories.py` — each factory returns a Pipeline containing preprocessing + estimator; grid keys ⊆ param names (no silent failures); XGB pipeline accepts DataFrame input.
4. **GREEN**: Implement `factories.py`.
5. **RED**: `test_nested_cv.py` on a tiny synthetic bundle (n=60, 2 classes): (a) number of FoldRecords == repeats × folds × models; (b) **leakage guard test** — assert best_params differ across outer folds when data is engineered so different folds favor different depths (proves tuning happens per-fold); (c) determinism — same base seed → identical record sequence; (d) every test fold seen exactly once per repeat.
6. **GREEN**: Implement `nested_cv.py` + `runner.py` (JSONL append, resume by skipping existing dataset+model files).
7. Dry-run on smallest dataset (synthetic, n=50) end-to-end; inspect one FoldRecord manually against a hand-run of `cross_validate` for the same split.
8. Commit red→green history.

## Success Criteria

- [ ] Leakage guard test passes (per-fold tuning proven)
- [ ] Determinism test passes (same seed → same records)
- [ ] Full synthetic-dataset dry run completes; JSONL well-formed
- [ ] Metrics module validated against analytic values

## Risk Assessment

- Nested CV compute blowup on Framingham/Stroke (50 outer × 3 inner × 16 combos): mitigation — grids kept small; `n_jobs=-1` in GridSearchCV only (outer loop serial for reproducible JSONL order); estimate runtime in dry run before full launch.
- XGBoost non-determinism across threads: mitigation — fix `n_jobs=1` inside estimator during CV runs if bitwise reproducibility required; document trade-off.
