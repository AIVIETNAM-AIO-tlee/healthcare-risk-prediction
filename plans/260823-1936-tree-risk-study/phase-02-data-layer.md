---
phase: 2
title: "Data Layer"
status: pending
priority: P1
effort: "2d"
dependencies: ["phase-01-research.md"]
---

# Phase 2: Data Layer

## Overview

TDD-build the dataset loading layer: one loader per dataset returning a uniform `(X: pd.DataFrame, y: pd.Series, metadata)` bundle, plus vitals feature aggregation for the synthetic dataset. Tests written first against tiny fixture files.

## Requirements

- Functional:
  - `load_dataset(slug) -> DatasetBundle` for 6 slugs; `DatasetBundle` = dataclass with `X`, `y`, `task_type` ("binary"|"multiclass"), `feature_names`.
  - Vitals aggregation: patient-level mean/std/min/max of HR, SBP, DBP, temp, SpO2 joined to demographics (≈25 features).
  - Deterministic output: same input file + slug → identical X, y (sorted columns, stable dtypes).
- Non-functional: loaders never mutate raw files; no target leakage (patient_id excluded from features).

## Architecture

```
src/tree_risk_stability/
├── data/
│   ├── base.py            # DatasetBundle dataclass, load_dataset registry
│   ├── synthetic_loader.py # demographics+vitals merge, aggregation
│   ├── public_loaders.py   # heart, breast_cancer, pima, framingham, stroke
│   └── vitals_features.py  # aggregation helpers
tests/data/
├── fixtures/               # miniature CSV/JSON copies (5-10 rows)
├── test_synthetic_loader.py
├── test_public_loaders.py
└── test_vitals_features.py
configs/experiments/<slug>.yaml
```

## Related Code Files

- Create: `src/tree_risk_stability/data/{base,synthetic_loader,public_loaders,vitals_features}.py`
- Create: `configs/experiments/*.yaml` (6 configs: slug, path, target column, drop columns, task type)
- Create: tests as listed above

## Implementation Steps

1. **RED**: Write `test_vitals_features.py` — hand-computed expected aggregates on a 2-patient × 3-hour fixture (mean/std/min/max per vital); property test: aggregation is invariant to row order.
2. **GREEN**: Implement `vitals_features.py` until tests pass.
3. **RED**: Write `test_synthetic_loader.py` — fixture-based: correct row count, label mapping Low/Medium/High preserved as ordered classes, no NaN in X, `patient_id` not among feature names, task_type == "multiclass".
4. **GREEN**: Implement `synthetic_loader.py`.
5. **RED**: Write `test_public_loaders.py` — parametrized over 5 datasets with mini-fixtures: schema check, target dtype, binary task flag; explicit cases: Stroke bmi-NaN retained (imputation happens later in pipeline), Pima zero-codes converted to NaN.
6. **GREEN**: Implement `public_loaders.py` + YAML config parsing (`load_config(slug)`).
7. Smoke-import integration test: `import xgboost, shap` succeeds on this platform (pins validated).
8. Run full pytest suite; commit red→green history in conventional commits.

## Success Criteria

- [ ] `pytest` green; ≥ 20 tests covering all loaders and aggregation
- [ ] `load_dataset(slug)` returns identical bundles across two calls (hash equality)
- [ ] All 6 experiment YAMLs parse and validate
- [ ] No feature column equals or derives from `patient_id`

## Risk Assessment

- Real-file vs fixture drift: mitigation — one integration test marked `@slow` that loads real files if present under `data/raw/`, skipped otherwise.
- Class-order instability in multiclass labels: mitigation — enforce fixed category order ["Low","Medium","High"] in loader, tested.
