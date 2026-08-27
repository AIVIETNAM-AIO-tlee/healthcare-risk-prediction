# Model Experiment Guide - AI Engineer (Model)

This module runs the three models used by the study: **AdaBoost, XGBoost, and LightGBM**.
It evaluates each model on each of the three processed healthcare datasets using the same
stratified 5-fold cross-validation protocol and then performs one final evaluation on the
untouched test split.

## 1. Get the real data files

The dataset CSVs are tracked by Git LFS. A GitHub source-code ZIP can contain only LFS
pointer text instead of the real CSV content. In a normal repository clone, run:

```bash
git lfs install
git lfs pull
```

## 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## 3. Run all experiments

From the repository root:

```bash
python -m src.experiments.run_models --config config.yaml
```

Run only selected datasets/models when debugging:

```bash
python -m src.experiments.run_models \
  --config config.yaml \
  --datasets dataset2 \
  --models adaboost xgboost lightgbm
```

## 4. Outputs

Results are written to `results/model_evaluation/`:

- `fold_metrics.csv`: one row per dataset/model/fold. This is the main input for RQ2.
- `cv_summary.csv`: mean, standard deviation, minimum, maximum, and range across folds.
- `test_metrics.csv`: final hold-out test metrics after refitting on the full development set.
- `model_comparison.csv`: within-dataset CV rank plus hold-out metrics.
- `overall_model_comparison.csv`: average within-dataset rank and unweighted cross-dataset means.
- `experiment_metadata.json`: Python/package versions and SHA-256 hash of `config.yaml`.

The implemented metrics are ROC-AUC, PR-AUC, Recall, and F1. PR-AUC is the primary ranking
metric because the healthcare datasets can be class-imbalanced. Recall and F1 use the fixed
0.50 decision threshold in `config.yaml`.

## 5. Reproducibility choices

- Random seed: `42`.
- Stratified 5-fold CV with shuffling.
- A new model object is created for every fold.
- Balanced sample weights are computed from the training portion of each fold only.
- The same fixed hyperparameter configuration is used across all datasets so fold/dataset
  stability is not confounded by a separate per-dataset hyperparameter search.
- XGBoost and LightGBM use conservative learning rates, subsampling, and regularization.
- LightGBM deterministic CPU options are enabled.

## 6. Important methodology note for the team

The current repository creates processed development/test CSVs before the model CV stage.
This protects the final test set from preprocessing leakage, but the preprocessing objects
were fitted once on the entire development split before that split is divided into five CV
folds. For the strictest possible CV protocol, imputers/encoders/scalers should eventually
be re-fitted inside each training fold. The current model code intentionally consumes the
team's existing processed files so this PR stays within the AI Engineer (Model) scope.
