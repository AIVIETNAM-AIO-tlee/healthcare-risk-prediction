# Healthcare Risk Prediction

Comparative research on AdaBoost, XGBoost, and LightGBM for binary healthcare risk prediction across three public datasets. The study evaluates predictive performance, predictive stability across cross-validation folds, and SHAP feature-ranking stability.

Repository: https://github.com/AIVIETNAM-AIO-tlee/healthcare-risk-prediction

## Research Questions

- **RQ1:** How do AdaBoost, XGBoost, and LightGBM compare across the healthcare datasets?
- **RQ2:** How stable is predictive performance across cross-validation folds?
- **RQ3:** How stable are SHAP-based feature-importance rankings across folds?

## Datasets

| Key        | Dataset                                         | Source                                                                                         | Raw file                                        |
| ---------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `dataset1` | Personal Key Indicators of Heart Disease (2022) | [Kaggle](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) | `heart_2022_with_nans.csv`                      |
| `dataset2` | Heart Failure Prediction                        | [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)                 | `heart.csv`                                     |
| `dataset3` | Heart Disease Health Indicators (BRFSS 2015)    | [Kaggle](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset)   | `heart_disease_health_indicators_BRFSS2015.csv` |

The raw files are expected under `data/raw/<dataset>/`. Review the original dataset pages for licensing, attribution, and usage conditions before redistribution.

## Project Structure

```text
config.yaml                         Experiment and preprocessing configuration
src/main.py                         Raw-data preprocessing entry point
src/config.py                       Dataset metadata and path registry
src/data/                           Loading, splitting, preprocessing, fold preparation
src/experiments/run_models.py       Model comparison and predictive evaluation
src/experiments/run_shap.py         SHAP explainability and ranking stability
src/evaluation/                     Metrics and ROC-curve plotting
notebooks/                          EDA and experiment analysis notebooks
data/raw/                           Raw datasets
 data/processed/                    Processed train/test files and fold manifests
results/model_evaluation/           Model metrics, rankings, and ROC plots
results/shap_explainability/       SHAP tables, metadata, and plots
docs/report/                        Research report and report assets
```

## Setup

Use Python 3.11 or newer. Install dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

The datasets are tracked with Git LFS in the full repository. If required:

```powershell
git lfs install
git lfs pull
```

## Run the Pipeline

Run these commands from the repository root.

### 1. Preprocess data

```powershell
python -m src.main
```

This creates, for each dataset:

- `data/processed/<dataset>/train.csv`
- `data/processed/<dataset>/test.csv`
- `data/processed/<dataset>/kfold_indices.csv`

The fold manifest is shared by model evaluation and SHAP analysis. Preprocessing is fitted on fold-training rows during CV and on the complete development split for the final hold-out evaluation.

### 2. Run model experiments

```powershell
python -m src.experiments.run_models --config config.yaml
```

For a smaller debugging run:

```powershell
python -m src.experiments.run_models `
  --config config.yaml `
  --datasets dataset2 `
  --models adaboost xgboost lightgbm
```

### 3. Run SHAP analysis

```powershell
python -m src.experiments.run_shap --config config.yaml
```

The SHAP runner produces fold-level feature importance, pairwise ranking stability, consensus rankings, and plots including beeswarm, bar, violin, ranking heatmap, and fold-correlation heatmap figures.

### 4. Use notebooks

Notebooks are intended for EDA, visualization, and result analysis. The reproducible pipeline entry points are the commands above; run them before opening the notebooks if processed data or result artifacts need to be regenerated.

## Configuration

`config.yaml` is the source of experiment-level settings:

- random seed, train/test ratio, and cross-validation;
- primary metric and decision threshold;
- class-balance policy;
- preprocessing thresholds;
- dataset paths and fold-manifest paths;
- model hyperparameters;
- SHAP output and sampling settings.

`src/config.py` stores dataset-specific metadata such as raw filenames, target columns, numeric columns, duplicate handling, and invalid-zero handling. `src/experiment_config.py` loads and validates the YAML configuration.

## Outputs

Model outputs are written to `results/model_evaluation/`:

- `fold_metrics.csv`
- `cv_summary.csv`
- `test_metrics.csv`
- `model_comparison.csv`
- `overall_model_comparison.csv`
- `roc_curves_dataset1.png`, `roc_curves_dataset2.png`, `roc_curves_dataset3.png`
- `roc_curves_all_datasets.png`
- `experiment_metadata.json`

SHAP outputs are written to `results/shap_explainability/`:

- `shap_feature_importance.csv`
- `shap_fold_stability.csv`
- `shap_stability_summary.csv`
- `shap_consensus_ranking.csv`
- `shap_execution_summary.csv`
- `shap_metadata.json`
- `plots/` for explainability and stability figures

## Evaluation Design

The experiment uses fixed configurations shared across datasets rather than dataset-specific hyperparameter tuning. Each development partition uses stratified five-fold cross-validation. ROC-AUC, PR-AUC, Recall, and F1-score are reported; PR-AUC is the primary ranking metric, while Recall and F1 use a fixed threshold of `0.50`.

SHAP explanations are computed on held-out validation samples. XGBoost and LightGBM use TreeSHAP; AdaBoost uses permutation SHAP because the pinned SHAP version does not support its scikit-learn estimator through TreeExplainer. Raw SHAP magnitudes are therefore not compared directly across model families; RQ3 focuses on within-model ranking stability across folds.

## Testing and Validation

Compile-check the Python source:

```powershell
python -m compileall -q src tests
```

Run the test suite after installing dependencies:

```powershell
python -m pytest -q
```

Build the report from `docs/report/` with the project's LaTeX toolchain.

## Collaboration

Project planning and Definition of Done are managed through Jira. Changes affecting datasets, preprocessing, models, metrics, experiments, or report claims should be linked to a Jira task and reviewed by the relevant team role. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for collaboration, review, dataset-provenance, and reporting expectations.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow, validation commands, pull-request checklist, and research reproducibility requirements.

## Report and Presentation

- Report source: [`docs/report/Research_Template_Report.tex`](docs/report/Research_Template_Report.tex)
- Demo video: https://youtu.be/a0pVMnRgGaE
- Presentation: https://drive.google.com/file/d/1foDtkD4MwgivJIOTOfVh2aazJ1p3nHq/view?usp=sharing
