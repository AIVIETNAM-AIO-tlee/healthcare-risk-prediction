---
phase: 4
title: "Explainability"
status: pending
priority: P1
effort: "2d"
dependencies: ["phase-03-model-pipeline.md"]
---

# Phase 4: Explainability

## Overview

TDD-build the SHAP layer: per-fold feature importance extraction via TreeExplainer and the four stability metrics for RQ3 (Kendall's W, pairwise Spearman ρ, Jaccard@k, mean|SHAP| coefficient of variation). Extends the runner to emit per-fold rankings alongside FoldRecords.

## Requirements

- Functional:
  - `extract_ranking(model, X_fold) -> pd.Series` (feature → mean|SHAP|, sorted desc); multiclass: mean over classes before abs... precisely: mean of per-class mean|SHAP| (document exact formula in docstring).
  - `kendalls_w(rankings: list[pd.Series]) -> float` in [0,1]; `pairwise_spearman(rankings) -> np.ndarray`; `jaccard_at_k(rankings, k) -> float`; `shap_cv(rankings) -> pd.Series` (per-feature CV).
- Non-functional: rankings stored as JSONL `{dataset, model, repeat, fold, ranking:{feature:value}}`; consistent feature ordering guaranteed by Phase 2 loaders.

## Architecture

```
src/tree_risk_stability/explain/
├── shap_extract.py    # TreeExplainer wrapper, multiclass aggregation
├── stability.py       # 4 stability metrics, pure functions on rankings
└── runner_patch.py    # extends Phase 3 runner loop with ranking extraction
results/raw/
├── <slug>__<model>__rankings.jsonl
```

## Related Code Files

- Create: `src/tree_risk_stability/explain/{shap_extract,stability}.py`
- Modify: `src/tree_risk_stability/models/nested_cv.py` (emit rankings)
- Create: `tests/explain/test_stability.py`, `tests/explain/test_shap_extract.py`

## Implementation Steps

1. **RED**: `test_stability.py` against textbook values:
   - Perfect agreement (k identical rankings) → Kendall's W = 1.0; fully reversed orderings → W = 0.
   - Spearman matrix diagonal = 1; symmetric; hand-computed 3-ranking example checked numerically.
   - Jaccard@k: two rankings sharing exactly 2 of top-5 → Jaccard = 2/8; identical → 1.0; disjoint → 0.0.
   - CV metric: constant SHAP values → CV = 0; property test — metrics invariant to common monotone value scaling.
2. **GREEN**: Implement `stability.py` (pure numpy/pandas; Kendall's W computed from mean rank deviations; handle ties via average ranks).
3. **RED**: `test_shap_extract.py` — tiny fitted tree on fixture data: ranking index == training feature names; all values ≥ 0; deterministic across two extractions; multiclass case returns single Series not array.
4. **GREEN**: Implement `shap_extract.py`.
5. **RED→GREEN**: integration test — run_nested_cv on tiny bundle produces one ranking line per fold record; ranking features match bundle features exactly.
6. Update methodology doc with the exact multiclass aggregation formula and tie-handling rules.

## Success Criteria

- [ ] All stability metrics pass analytic-value tests
- [ ] Ranking extraction deterministic; multiclass aggregation documented
- [ ] Runner emits paired records+rankings JSONL without breaking resume logic
- [ ] pytest suite green

## Risk Assessment

- SHAP interventional vs tree_path_dependent perturbation choice changes values: mitigation — pin `TreeExplainer(model, data=background, feature_perturbation="interventional")` with train-fold background sample (cap 200 rows), record in methods.
- Tie-heavy small grids make Kendall's W conservative: mitigation — ties handled by average-rank method, noted as limitation in paper discussion.
