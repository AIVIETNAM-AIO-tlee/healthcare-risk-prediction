# [QA] Preprocessing / Split / Stratified K-Fold Validation — Findings

> Date: 2026-08-26. Scope: correctness + reliability of preprocessing, dataset
> splitting, and Stratified K-Fold CV across the 3 healthcare datasets.
> Result: **65/65 tests pass.** 1 real bug found and fixed, 2 test-design bugs
> fixed during authoring.

## What was validated

Pipeline per dataset: `load_raw_dataset` → dedupe (D3) → zero-as-missing (D2)
→ `clean_target` → stratified 80/20 split → fit preprocessor on train only →
transform test with the same fitted object → Stratified K-Fold on train split.

## Findings

### QA-BUG-1 — Single-category column passed through as raw string (FIXED)

`src/data/preprocessing.py`: `multi_columns` was selected with `nunique() > 2`,
so a categorical column with exactly one training value fell into neither the
binary path nor the one-hot path. It survived both `fit_preprocessor` and
`transform_features` as a raw string column — non-numeric data reaching model
input, and unseen values at transform time silently injected new strings.
Fix: `nunique() != 2` routes single-category columns through one-hot encoding.
Regression test: `tests/test_preprocessing.py::test_single_category_column_is_one_hot_encoded_not_passed_through`.
No current dataset triggers it (all categoricals ≥2 values), so processed CSVs
in `data/processed/` are unaffected; fix protects future configs.

### QA-BUG-2 — K-Fold guard rejected valid binary targets (FIXED, same commit)

Initial `stratified_kfold_splits` guard required `nunique() >= n_splits`
classes — wrong requirement: stratified folds need `n_splits` samples *per
class*, not classes. Binary targets (all 3 datasets) would have raised.
Fix: guard removed; sklearn's own minority-class validation kept.

### QA-OK — Leakage probes all pass

Fitted on train only, verified against full-data statistics (probes fail if
fitting had used full data):
- Imputer medians == train medians ≠ full-data medians (`test_imputer_median_fitted_on_train_only`)
- Scaler means == post-imputation train means ≠ full-data means (`test_scaler_fitted_on_train_only`)
- Categorical modes == train modes, differ from full where distinguishable (`test_categorical_modes_fitted_on_train_only`)
- Test split transformed exclusively via stored `FittedPreprocessor` statistics — no re-fitting path exists in `transform_features`.

### QA-OK — Separation & CV properties

- Train/test: sizes sum to total, stratification holds, seed-reproducible, different under different seeds.
- K-Fold (5 folds, shuffle=True, seed 42): every training sample validated exactly once; train/val indices disjoint within folds; class proportions preserved (<0.02 deviation); test split entirely outside the fold loop.
- Target appears exactly once, as leading column; never among input features.
- Processed outputs NaN-free; train/test share identical feature layout.
- Raw CSVs byte-identical after pipeline runs (checksum test).
- Full pipeline (split + preprocess + transform) bit-identical across repeated runs for all 3 datasets.
- Dataset1 survey duplicates can legitimately straddle splits (only D3 drops duplicates) — overlap bounded by raw duplicate count, asserted in integration test.

## Test suite

| File | Coverage |
|---|---|
| `tests/test_preprocessing.py` | Missing-value handling (numeric median, categorical mode), binary sorted-mapping encoding, one-hot expansion, scaling moments, feature dims, input immutability, determinism, invalid inputs, transform fallbacks |
| `tests/test_split.py` | Split ratio/stratification/disjointness/reproducibility/mutation-safety; K-Fold count/partition/no-overlap/proportions/seed behavior/minority-class validation |
| `tests/test_integration_real_datasets.py` | All 3 real datasets end-to-end: leakage probes, separation, dims, NaN-freedom, raw-file immutability, CV-excludes-test, repeat-run consistency |

Run: `.venv/bin/python -m pytest tests -q` → **65 passed** (~40s; integration
suite reads the raw CSVs once).

Environment note: repo `requirements.txt` is empty; `.venv` created via uv
(Python 3.12, sklearn 1.9.0, pandas 3.0.5, numpy 2.5.2, pytest). Populate
`requirements.txt` before sharing the environment.

## Out of scope / open items (from qa handoff F1–F8)

Class-imbalance policy (F1), threshold policy (F2), statistical tests (F3),
hyperparameter protocol (F4), IQR outliers — not implemented anywhere, nothing
to validate (F5), SHAP-AdaBoost spike (F6). CV runner consuming
`stratified_kfold_splits` is still to be built; helper + tests are ready for it.
