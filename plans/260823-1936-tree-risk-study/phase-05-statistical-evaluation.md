---
phase: 5
title: "Statistical Evaluation"
status: pending
priority: P2
effort: "2d"
dependencies: ["phase-04-explainability.md"]
---

# Phase 5: Statistical Evaluation

## Overview

TDD-build the inference layer and execute the full experiment matrix (6 datasets × 3 models × 50 folds). Produces results tables for RQ1/RQ2/RQ3 plus Friedman/Nemenyi, Wilcoxon-Holm, Brown-Forsythe outputs.

## Requirements

- Functional:
  - `friedman_nemenyi(score_matrix) -> {chi2, p, posthoc_p}` — models as columns, datasets×repeats as rows.
  - `wilcoxon_holm(model_a_scores, model_b_scores, family) -> {statistic, p_raw, p_adj}` with Holm step-down over the 3 model pairs.
  - `brown_forsythe(variance_groups)` for RQ2 variance comparison.
  - Aggregation: per dataset+model → mean±std, CV%, IQR of each metric; stability summaries per dataset+model → W, mean pairwise ρ, Jaccard@5/10, median SHAP-CV.
- Non-functional: every table reproducible from raw JSONL by one command; no manual spreadsheet steps.

## Architecture

```
src/tree_risk_stability/evaluation/
├── inferential.py     # friedman/nemenyi/wilcoxon-holm/brown-forsythe wrappers (scipy + custom nemenyi critical difference)
└── aggregate.py       # JSONL → tidy DataFrames → results/tables/*.csv
results/tables/        # rq1-performance.csv, rq2-stability.csv, rq3-shap-stability.csv, tests-*.csv
```

## Related Code Files

- Create: `src/tree_risk_stability/evaluation/{inferential,aggregate}.py`
- Create: `tests/evaluation/test_inferential.py`, `tests/evaluation/test_aggregate.py`
- Modify: none upstream

## Implementation Steps

1. **RED**: `test_inferential.py` — Friedman on published example dataset (known χ² and p from textbook); Nemenyi CD formula checked against Demšar (2006) critical values table (k=3, α=0.05 → CD ≈ 0.449·√(k(k+1)/6N) form verified numerically); Holm correction on hand case: smallest raw p × m ordering enforced; property test — Wilcoxon of identical samples returns p=1 not NaN.
2. **GREEN**: Implement `inferential.py`.
3. **RED**: `test_aggregate.py` — synthetic JSONL fixture with known means/stds → aggregated CSV matches expected values; missing-fold detection raises; multiclass and binary slugs coexist in one table.
4. **GREEN**: Implement `aggregate.py` + CLI (`python -m tree_risk_stability.evaluation.aggregate --results-dir results/raw --out results/tables`).
5. Launch full experiment run (background, ~<3h): all 6 datasets × 3 models. Monitor, then re-run aggregate.
6. Generate figures: performance distributions (box/violin per dataset), CV% heatmap, Kendall's W heatmap, Jaccard@k curves, SHAP top-k rank-flip plots, critical difference diagrams (RQ1 AUC and F1 separately).
7. Sanity-review pass: do synthetic-dataset numbers show expected high variance? Does XGBoost dominate large datasets? Flag anomalies before writing phase.

## Success Criteria

- [ ] Inference functions validated against published examples
- [ ] Full experiment matrix complete; zero failed folds in logs
- [ ] All CSVs + figures regenerate from raw via single commands
- [ ] Anomaly review documented in `plans/<dir>/reports/run-review.md`

## Risk Assessment

- Runtime overrun on personal machine: mitigation — Phase 3 dry-run estimate gates full launch; option to reduce repeats to 5 with explicit note (protocol change documented, not silent).
- Multiple-comparison criticism (50 correlated folds inflate significance): mitigation — report effect sizes alongside p-values; discuss fold-correlation limitation explicitly in paper; consider per-repeat aggregation as robustness check.
