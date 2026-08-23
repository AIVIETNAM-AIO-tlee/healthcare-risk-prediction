---
phase: 1
title: "Research"
status: pending
priority: P2
effort: "1d"
dependencies: []
---

# Phase 1: Research

## Overview

Consolidate methodological decisions and dataset acquisition knowledge into `docs/` so later phases implement against written specs, not memory. No package code yet except project scaffolding.

## Requirements

- Functional: dataset acquisition plan verified by actually downloading all 6 datasets; metric formulas documented with references; hyperparameter grids fixed.
- Non-functional: all docs in English; every numeric decision traceable to brainstorm report or literature.

## Architecture

Deliverables are documents + raw data cache:

```
docs/
├── research-methodology.md      # CV protocol, leakage rules, metric definitions
├── datasets-catalog.md          # per-dataset: source URL, schema, target, license, known issues
data/raw/<dataset-slug>/         # immutable downloaded copies
pyproject.toml                   # package scaffold: deps pinned (scikit-learn, xgboost, shap, scipy, pandas, pyyaml)
```

## Related Code Files

- Create: `docs/research-methodology.md`, `docs/datasets-catalog.md`, `pyproject.toml`, `.gitignore`, `README.md`
- Create: `scripts/download-datasets.sh` (idempotent download of public CSVs)

## Implementation Steps

1. Scaffold repo: `pyproject.toml` (src-layout, dev extras pytest+hypothesis), `.gitignore` (`data/raw/`, `results/raw/`, notebooks checkpoints), minimal `README.md`.
2. Download and verify all 6 datasets into `data/raw/`; record SHA256 checksums in catalog doc.
3. Write `docs/datasets-catalog.md`: for each dataset — columns, dtypes, target definition, class balance, missing-value pattern, license, preprocessing notes (e.g., Stroke `bmi` NaNs; Pima zero-as-missing codes for glucose/BMI).
4. Write `docs/research-methodology.md`: Repeated Stratified 5×10 protocol; inner 3-fold grid search; per-task metric mapping (binary ROC-AUC vs multiclass macro-OVR); SHAP aggregation rule (mean|SHAP| over classes); seeds policy (base seed 42, repeat seeds derived).
5. Fix hyperparameter grids per model (DT: depth/split/min-leaf ~8 combos; RF: n-estimators/max-depth/max-features ~12; XGB: lr/depth/n-estimators/subsample ~16) — document rationale.
6. Verify synthetic vitals aggregation spec: groupby patient → mean/std/min/max × {HR, SBP, DBP, temp, SpO2} + demographics → write feature list into methodology doc.

## Success Criteria

- [ ] All 6 datasets present under `data/raw/` with recorded checksums
- [ ] Catalog covers schema + target + missing values for every dataset
- [ ] Methodology doc defines CV scheme, grids, metrics, seeds unambiguously
- [ ] `pip install -e ".[dev]"` works; `pytest` collects 0 tests without error

## Risk Assessment

- Public dataset URLs rot (Framingham mirrors vary): mitigation — pin exact mirror URLs in catalog + keep local copies out of git but checksummed.
- XGBoost/shap version incompatibility on macOS arm64: mitigation — pin versions in `pyproject.toml`, smoke-import test in Phase 2.
