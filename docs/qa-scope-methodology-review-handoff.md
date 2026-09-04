# [QA] Project Scope & Methodology Review — Handoff

> Date: 2026-08-26. Status: review done, waiting to be actioned. No code written yet.
> Inputs: report RQs (`docs/report/Research_Template_Report.tex`), current pipeline (`src/`), team workflow diagram (Stratified 5-Fold CV on the 80% train split, metrics F1/ROC-AUC/PR-AUC/Recall, SHAP global/local + stability via Kendall τ / Spearman / Top-K Jaccard).

## Background

- Project: predict heart disease risk, compare AdaBoost / XGBoost / LightGBM across 3 Kaggle datasets.
- RQ1: compare performance of the 3 models. RQ2: how stable performance is across folds. RQ3: how stable SHAP feature rankings are across folds.
- Already done: leakage-safe preprocessing (fit on train only), 80/20 split, config for 3 datasets.
- Not done yet: CV runner, model training, evaluation metrics, SHAP, stability analysis, app, tests.

## Main conclusion

Scope is sound and the RQs are well-formed. The diagram covers only half the problem. The other half: **class imbalance is ignored** and **stability/statistics are not defined well enough**. Fix these before running experiments — fixing them after means re-running everything.

## Findings to fix (in priority order)

### F1 — No class imbalance policy (CRITICAL)
- Positive rate: D1 ~5%, D3 ~10%, D2 ~50%.
- Pick one policy and use it for all 3 models: `class_weight="balanced"` or resampling (SMOTE / undersampling). Without it, RQ1 rankings will be biased toward the majority class.
- Action: decide the policy, add it to §Data & Experimental Design + `src/`.

### F2 — Threshold for F1/Recall is undefined (CRITICAL)
- The default 0.5 threshold means nothing when positives are ~5%.
- Options: (a) fixed 0.5 threshold + state it as a limitation; (b) tune the threshold inside each fold; (c) use AUC as the primary metric, F1/Recall as secondary.
- Action: pick one, state it clearly in the report.

### F3 — Missing statistical tests for RQ1/RQ2 (HIGH)
- Comparing 3 models without a statistical test means "A is better than B" has no basis.
- Proposal: Friedman test on per-fold results + post-hoc Nemenyi. Requirement: all 3 models must use the same folds (shared folds) so the pairing is valid.
- Action: add to experimental design; make sure the CV runner saves per-fold results in (model × fold, same seed) pairs.

### F4 — Hyperparameter protocol not stated (HIGH)
- Tuning inside each fold → RQ2 measures tuning noise, not model instability.
- Tuning on all data → leaks into the test set.
- Proposal for this scope: fixed, sensible default hyperparameters for every model, stated as a limitation. Nested CV is correct but overkill here.
- Action: fix defaults, list the values in a report appendix.

### F5 — IQR outlier removal should only apply to D2 (MEDIUM)
- D1/D3 are survey data, mostly binary/discrete; IQR clipping cuts valid tail values (high BMI is a real disease signal).
- D2 has continuous clinical variables → IQR is justified there.
- If applied: compute IQR bounds on train only, transform test with those bounds (avoid leakage).
- Note: tree models are robust to outliers — this step is optional, method not mandated.
- Action: decide "D2 only" or drop it entirely; if kept, add to `src/data/preprocessing.py` as fit-on-train.

### F6 — SHAP for AdaBoost is a technical risk (HIGH)
- `shap.TreeExplainer` works natively for XGBoost/LightGBM; sklearn AdaBoost (SAMME stumps) has no native path.
- Options: KernelExplainer on a sample subset, or aggregate SHAP per stump.
- If this dies → RQ3 scope must change now, not after the experiments run.
- Action: spike it on the smallest dataset (918 rows, `data/raw/dataset2/heart.csv`) before committing to the design.

### F7 — Dataset numbering mismatch between diagram and repo (LOW)
- Diagram: D2=250k, D3=918. Repo: dataset2=918 (`heart.csv`, Heart Failure Prediction), dataset3=250k (BRFSS 2015).
- Action: pick one convention (recommend the repo's), sync diagram + report + `src/config.py`.

### F8 — Methodology + Data & Experimental Design sections in the report are empty (HIGH)
- Every decision F1–F7 must land there BEFORE experiments run — write it first, like a pre-registration, so results can't be accused of cherry-picking.
- Action: write §Methodology + §Data & Experimental Design right after deciding F1–F7.

## What is already good (keep as is)

- Leakage-safe preprocessing: imputer/encoder/scaler fitted on train only, reused via `FittedPreprocessor`.
- Explicit column typing instead of dtype inference.
- Per-dataset quirk handling (duplicates in D3, zero-as-missing in D2).
- Seeded reproducibility (`RANDOM_STATE=42`).

## Suggested order of work when resuming

1. Decide policies F1+F2+F4 → write into §Data & Experimental Design.
2. Fix the statistical tests: Friedman+Nemenyi (RQ1/RQ2), Kendall τ + Top-K Jaccard k=10 (RQ3) — F3.
3. SHAP-AdaBoost spike on dataset2 (918 rows) — F6.
4. Decide IQR: D2 only or drop — F5.
5. Sync dataset numbering — F7.
6. Write §Methodology + §Data & Experimental Design — F8.
7. Only then build the code: CV runner (shared folds), models/, evaluation/, explainability/.
