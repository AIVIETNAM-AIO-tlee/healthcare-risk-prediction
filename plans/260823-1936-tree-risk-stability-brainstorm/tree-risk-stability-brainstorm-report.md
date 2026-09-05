---
title: "Brainstorm Report: Tree-based Models for Healthcare Risk Prediction"
date: 2026-08-23
type: brainstorm-report
status: approved
---

# Brainstorm Report: A Comparative Study of Tree-based Models for Healthcare Risk Prediction — Cross-validation and SHAP-based Stability Analysis

## 1. Problem Statement & Requirements

### Research Questions

- **RQ1**: How do Decision Tree, Random Forest, and XGBoost compare in healthcare risk prediction across multiple datasets?
- **RQ2**: How stable is the predictive performance of these models across cross-validation folds?
- **RQ3**: How stable are SHAP-based feature importance rankings across cross-validation folds?

### Confirmed Requirements

| Item | Decision |
|---|---|
| Goal | Publishable research paper |
| Datasets | 6 datasets (1 synthetic + 5 public) |
| Targets | Native per-dataset targets (synthetic = multiclass Low/Med/High; public = binary) under a unified evaluation protocol |
| Features | Synthetic dataset: demographics + aggregated vitals; public datasets: native features |
| CV scheme | Repeated Stratified K-Fold, nested CV with inner hyperparameter tuning |
| SHAP stability metrics | Kendall's W + pairwise Spearman ρ + Jaccard@k + variance analysis of mean\|SHAP\| |
| Statistics | Friedman + Nemenyi post-hoc, Wilcoxon signed-rank with Holm correction |
| Deliverables | Reproducible package + notebooks + report/paper draft |
| Language | English for all docs and reports |
| Timeline | 2–4 weeks |

## 2. Codebase Context (Scout Findings)

- Workspace `Module3` is an empty git repo (fresh start).
- Dataset source repo `Santy3298/healthcare-ai-risk-dataset` contains:
  - `demographics.csv` — 50 patients × 6 columns (age, gender, smoking, diabetes, hypertension)
  - `disease_risk_labels.csv` — labels Low=30 / Medium=15 / High=5
  - `vitals_time_series.csv` — 1,200 rows = 50 patients × 24 hourly records (HR, SBP, DBP, temperature, SpO2)
  - `ehr_records.json`, `data_dictionary.xlsx`, license CC BY 4.0

## 3. Evaluated Approaches

### 3.1 Dataset Strategy

| Option | Verdict | Rationale |
|---|---|---|
| Single synthetic dataset only | Rejected | RQ1 requires "multiple datasets"; n=50 with 5 High samples makes stability estimates noise-dominated |
| Add public datasets (chosen) | Accepted | Diversity of scale (303–5.1k), classic benchmarks, strengthens "multiple datasets" claim |
| Synthetic augmentation only | Rejected | Generated data cannot substitute real-world distribution shift evidence |

Final pool: Synthetic repo (50) · UCI Heart Disease (303) · Breast Cancer Wisconsin (569) · Pima Diabetes (768) · Framingham (~4.2k) · Stroke Prediction (~5.1k)

### 3.2 Target Definition

User initially requested forcing multiclass everywhere. **Challenged and revised**: public healthcare datasets have inherently binary outcomes; fabricating 3-class labels would be arbitrary label construction and desk-rejection risk. Agreed resolution: native per-dataset targets + unified protocol (same CV scheme, same metric family per task type, same SHAP stability procedure).

### 3.3 Architecture

| Option | Verdict | Rationale |
|---|---|---|
| A. Pure notebooks | Rejected | 6× code duplication, not reproducible, weak for publication |
| B. Modular package + YAML-config experiments (chosen) | Accepted | DRY, reproducible, seeds fixed, scales to 6 datasets × 3 models × 50 folds |
| C. Existing AutoML frameworks (PyCaret etc.) | Rejected | No control over nested-CV internals or per-fold SHAP extraction |

## 4. Final Design

### Pipeline (per outer fold)

1. Preprocessing inside sklearn `Pipeline` (median impute + encode fit on train fold only → zero leakage)
2. Inner 3-fold grid search on train fold (small grids ~12–16 combos/model)
3. Fit best model → predict outer test fold → collect metrics
4. SHAP `TreeExplainer` → mean\|SHAP\| ranking (multiclass: aggregate over classes)

Repeated Stratified 5-Fold × 10 repeats = 50 measurement points per model/dataset.

### Metrics

- **RQ1**: ROC-AUC (macro-OVR for multiclass), weighted-F1, balanced accuracy, MCC
- **RQ2**: mean ± std, coefficient of variation, IQR; Brown–Forsythe test for variance comparison between models
- **RQ3**: Kendall's W across 50 rankings, pairwise Spearman ρ, Jaccard@5/@10, CV of mean\|SHAP\| values
- **Inference**: Friedman + Nemenyi post-hoc across models, Wilcoxon pairs + Holm correction, critical difference diagram (Demšar)

### Repository Layout

```
Module3/
├── configs/experiments/*.yaml      # one config per dataset
├── src/tree_risk_stability/
│   ├── data/                       # loaders, vitals aggregation
│   ├── models/                     # factories, grids, nested-CV runner
│   ├── explain/                    # SHAP extraction, stability metrics
│   ├── evaluation/                 # metrics, Friedman/Wilcoxon
│   └── visualization/plots.py
├── tests/
├── notebooks/                      # 01-eda, 02-results-analysis
├── data/{raw,processed}/
├── results/{raw,tables,figures}/
├── docs/  plans/  paper/
```

### Novelty Positioning (literature scan)

Related work uses Kendall's τ + Overlap@k for SHAP stability (IELF 2025; IEEE Access 2024 endocrinology; Alzheimer's 2026 multi-level stability framework). Gap: no systematic comparison of dual stability (performance + explanation) across the three canonical tree models over multiple healthcare datasets with repeated nested CV and full inferential statistics.

## 5. Implementation Considerations & Risks

- Synthetic dataset (n=50, High=5): noisy RQ2/RQ3 results expected — treat as a finding (stability degrades with sample size), frame as contribution.
- Stroke dataset imbalance (~1:19): stratification mandatory; add PR-AUC as supplementary metric.
- Multiclass SHAP aggregation choice must be documented in methods.
- Compute estimate: < 3 h total on personal machine for full run.

## 6. Success Criteria

- [ ] All 6 datasets load through tested loaders with documented schemas
- [ ] Full nested-CV run completes with fixed seeds, fully reproducible from one command
- [ ] Results tables answer RQ1–RQ3 with descriptive + inferential statistics
- [ ] Critical difference diagram + stability heatmaps produced as publication-ready figures
- [ ] Paper draft sections: intro, related work, methods, results, discussion

## 7. Next Steps

1. `/ck:plan` to produce phased implementation plan (recommended mode: default)
2. Phase order: package skeleton → loaders+tests → pipeline runner → SHAP module → stats module → runs → analysis notebooks → paper draft

## Unresolved Questions

None — all five discovery gates resolved (output, acceptance criteria, scope, constraints, touchpoints).
