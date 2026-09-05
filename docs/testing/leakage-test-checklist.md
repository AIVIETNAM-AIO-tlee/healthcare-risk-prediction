# Leakage Test Checklist — Traceability

QA RED suite guarding the study against preprocessing/CV/SHAP leakage. Tests are
written BEFORE implementation (plan mode `--tdd`): each currently fails with
`NotImplementedError` and turns GREEN when the owning phase implements it.

**Status legend:** `RED` = awaiting implementation · `GREEN` = implemented and passing
Update Status as phases complete. A test that goes GREEN then RED again is a regression — stop and fix before continuing.

## Leakage vectors → tests

| # | Vector | Risk if violated | Test file | Test | Owning phase | Plan ref | Status |
|---|--------|------------------|-----------|------|--------------|----------|--------|
| 1 | Imputer/encoder statistics fit on full data instead of train fold | Test-fold distribution informs preprocessing; metrics inflated | `tests/data/test_leakage_preprocessing.py` | `TestImputerTrainFoldOnly::test_imputer_statistics_come_from_train_only` (+ control `test_full_fit_leaks_when_done_wrong`) | Phase 3 (pipeline builders) | phase-03 Requirements "preprocessing inside Pipeline" | RED |
| 2 | Identifier / target columns reach model features | Direct target leakage; ID memorization | `tests/data/test_feature_hygiene.py` | `test_bundle_shape_contract`, `_assert_no_identifier_features` + `_assert_target_not_in_features` via `test_all_registered_loaders_exclude_identifiers_and_target` | Phase 2 (loaders) | phase-02 Requirements, Success Criteria "No feature column … patient_id" | RED |
| 3 | Inner grid search sees outer-test rows | Hyperparameters tuned on test data; optimistic RQ1/RQ2 | `tests/models/test_leakage_cv.py` | `TestInnerSearchSeesTrainOnly::test_inner_fits_never_touch_test_rows` (+ vacuity guard `test_spy_actually_fits_something`) | Phase 3 (`nested_cv.py`) | phase-03 step 5(b) leakage guard | RED |
| 4 | Singleton minority strata in INNER folds (synthetic regime) | Grid search decided by 1-sample validation; noise masquerading as model selection | `tests/models/test_leakage_cv.py` | `TestInnerFoldMinorityStrata::test_build_inner_cv_exists_and_yields_valid_splits` (+ `test_default_binary_config_uses_plain_stratification`) | Phase 3 (`build_inner_cv` override) | phase-03 Requirements inner-CV override | RED |
| 5 | Cross-patient statistics in vitals aggregation | One patient's features shift when another's rows change — group contamination | `tests/data/test_leakage_preprocessing.py` | `TestVitalsAggregationIsolation::test_patient_aggregates_independent_of_other_patients` (+ row-order invariance) | Phase 2 (`vitals_features.py`) | phase-02 vitals aggregation spec | RED |
| 6 | SHAP background includes test-fold rows | Explanations absorb test information; RQ3 rankings contaminated | `tests/explain/test_leakage_shap.py` | `TestBackgroundProvenance::test_signature_exposes_background_parameter`, `test_background_changes_attributions` | Phase 4 (`shap_extract.py`) | phase-04 interventional pin w/ train-fold background | RED |

## Supporting fixtures

- `tests/conftest.py` — `make_binary_bundle` (n=60, ID column present), `make_multiclass_mini` (Low/Medium/High = 6/3/1 synthetic regime). Dependency-free (numpy/pandas/sklearn only) so the suite runs before xgboost/shap pins exist.

## QA-mandated seams (new API required from implementation phases)

These exist ONLY to make leakage auditable. Phase owners must keep them public:

1. `models.nested_cv.make_spy_fit_context(train_idx, estimator_class)` — Vector 3
2. `models.nested_cv.build_inner_cv(y, config)` — Vector 4
3. `explain.shap_extract.extract_ranking(model, X_fold, background=None)` — explicit background parameter, Vector 6

## Out of scope here (covered by plan elsewhere)

- Outer-fold coverage/determinism/resume: `phase-03` step 5(c,d,e) tests
- Metric analytic values incl. singleton-positive case: `phase-03` step 1
- Aggregate count/pairing hard-fail: `phase-05` step 3
