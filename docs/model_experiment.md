# Model Experiment Guide - AI Engineer (Model)

The model experiment compares **AdaBoost, XGBoost, and LightGBM** on three processed healthcare datasets using a common stratified 5-fold cross-validation protocol and one final untouched hold-out test.

## Recommended experiment entry point: notebook

The primary experiment documentation is:

```text
notebooks/model_experiments.ipynb
```

The notebook contains the experiment setup, configuration, execution, result tables, and analysis for RQ1 and RQ2. Reusable model construction, evaluation metrics, configuration loading, and experiment-running logic remain in `src/`.

To reproduce the notebook from a fresh clone, first retrieve the Git LFS datasets and install dependencies:

```bash
git lfs install
git lfs pull
python -m pip install -r requirements.txt
```

Then open `notebooks/model_experiments.ipynb` in Jupyter or VS Code and run all cells. The notebook calls the reusable experiment runner in `src.experiments.run_models` and writes the same result files to `results/model_evaluation/`.

## Optional command-line execution

The experiment can also be run from the repository root without the notebook:

```bash
python -m src.experiments.run_models --config config.yaml
```

Selected datasets or models can be used for debugging:

```bash
python -m src.experiments.run_models \
  --config config.yaml \
  --datasets dataset2 \
  --models adaboost xgboost lightgbm
```

## Outputs

Results are written to `results/model_evaluation/`:

- `fold_metrics.csv`: one row per dataset/model/fold; main input for RQ2.
- `cv_summary.csv`: mean, standard deviation, minimum, maximum, and range across folds.
- `test_metrics.csv`: final hold-out test metrics after refitting on the full development set.
- `model_comparison.csv`: within-dataset CV rank plus hold-out metrics.
- `overall_model_comparison.csv`: average within-dataset rank and unweighted cross-dataset means.
- `shap_fold_importance.csv`: mean absolute SHAP importance for every feature in every validation fold.
- `shap_stability.csv`: cross-fold explanation stability using mean Kendall's tau, Spearman's rho, and Top-K Jaccard similarity.
- `experiment_metadata.json`: Python/package versions and SHA-256 hash of `config.yaml`.

The implemented metrics are ROC-AUC, PR-AUC, Recall, and F1. PR-AUC is the primary ranking metric because the healthcare datasets can be class-imbalanced. Recall and F1 use the fixed 0.50 decision threshold from `config.yaml`.

## SHAP stability

For each validation fold, the fitted model explains that fold's validation rows. Feature importance is the mean absolute SHAP value across those rows. The resulting feature rankings are compared across every pair of folds:

- Kendall's tau and Spearman's rho compare the full feature rankings.
- Top-K Jaccard measures overlap among the most important features; `top_k` is configured under `experiment.shap` and defaults to 10.

The untouched test split is not used for SHAP stability. Higher values indicate more stable explanations.

## Reproducibility choices

- Random seed: `42`.
- Stratified 5-fold CV with shuffling.
- A new model object is created for every fold.
- Balanced sample weights are computed from the training portion of each fold only.
- The same fixed hyperparameter configuration is used across all datasets.
- XGBoost and LightGBM use conservative learning rates, subsampling, and regularization.
- LightGBM deterministic CPU options are enabled.

## Methodology note

The repository currently provides processed development/test CSVs before the model CV stage. This protects the final test split from preprocessing leakage, but the preprocessing objects are fitted once on the entire development split before that split is divided into five CV folds. A stricter future protocol could re-fit preprocessing components inside each training fold. The model module intentionally consumes the team's existing processed files so that the implementation stays within the AI Engineer (Model) scope.
