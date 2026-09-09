# Jira Work Summary by Team Member

This document summarizes the work recorded in the Jira exports for the **AIO-CONQUER MODULE 3** project. All issues listed in the exports were marked **Done** as of 8 September 2026.

## Le Quang Thanh

**Role:** Team Leader and Tech Lead

Le Quang Thanh coordinated the project, established the development workflow, led the project structure and documentation, and completed the final review activities.

### Completed work

- **ACM3-1 - Get used to GitHub/Jira:** Read the project documentation and learned the basic GitHub and Jira workflows.
- **ACM3-2 - Project initiation, team alignment, and research definition:** Managed the project initiation epic and aligned the team around the research direction.
- **ACM3-3 - Kick-off meeting:** Scheduled meetings, defined working procedures, and assigned team roles.
- **ACM3-4 - Jira setup:** Created and configured the Jira project and defined the project work types.
- **ACM3-5 - GitHub setup:** Created and configured the GitHub repository and established the folder structure.
- **ACM3-19 - Introduction and research definition:** Wrote the report introduction and research questions, and defined the Definition of Done and task descriptions.
- **ACM3-17 - Results, report, and final review:** Coordinated the results, reporting, and final review epic.
- **ACM3-34 - Research architecture and general pipeline:** Wrote the research architecture and general machine-learning pipeline sections.
- **ACM3-35 - Presentation preparation:** Prepared the presentation structure and research overview.
- **ACM3-37 - Final review:** Conducted the final review of the report, source code, and documentation.

## Le Thanh Lam

**Role:** Data / AI Engineer

Le Thanh Lam was responsible for dataset collection and validation, exploratory data analysis, data preprocessing, feature preparation, and the data-related sections of the report.

### Completed work

- **ACM3-6 - Dataset collection and reliability checks:** Collected the three selected healthcare datasets, verified their provenance, schema, target variables, integrity, licensing conditions, limitations, and suitability for the research scope.
- **ACM3-9 - Data understanding and preparation:** Led the data understanding and preparation epic.
- **ACM3-12 - Exploratory data analysis:** Performed exploratory data analysis on the healthcare datasets.
- **ACM3-13 - Initial preprocessing:** Implemented missing-value handling, outlier handling, categorical encoding, and feature scaling for the three datasets.
- **ACM3-23 - Enhanced preprocessing:** Extended preprocessing to address class imbalance, outliers, and feature reduction.
- **ACM3-29 - Dataset description and EDA report:** Wrote the dataset description and exploratory data analysis sections of the report.
- **ACM3-30 - Data preprocessing methodology:** Documented the data preprocessing methodology used in the study.

## Pham Minh Dang Tran

**Role:** Model / AI Engineer

Pham Minh Dang Tran selected and implemented the tree-based models, defined experiment settings and evaluation metrics, implemented SHAP explainability, and wrote the model-related report sections.

### Completed work

- **ACM3-11 - Tree-based model selection:** Reviewed and selected the tree-based models for the project.
- **ACM3-10 - Machine-learning pipeline and model comparison:** Led the model-comparison epic.
- **ACM3-18 - Model parameters and experimental design:** Selected model parameters and hyperparameters, and wrote the related-work and experimental-design sections of the report.
- **ACM3-20 - Model implementation:** Implemented AdaBoost, XGBoost, and LightGBM.
- **ACM3-21 - Evaluation metrics and comparison:** Implemented ROC-AUC, PR-AUC, Recall, and F1 evaluation metrics and prepared model comparisons.
- **ACM3-22 - Reproducible experiment configuration:** Created `config.yaml` for reproducible model experiments.
- **ACM3-25 - SHAP explainability:** Implemented SHAP-based model explainability.
- **ACM3-32 - SHAP methodology:** Wrote the SHAP explainability methodology section of the report.

## Tan Du

**Role:** QA / Reviewer

Tan Du reviewed the research scope and methodology, validated the data and cross-validation workflow, and implemented tests for preprocessing, models, results, explainability, and SHAP stability.

### Completed work

- **ACM3-8 - Scope and methodology review:** Reviewed the project scope and research methodology.
- **ACM3-15 - Preprocessing and cross-validation validation:** Validated preprocessing and checked for cross-validation leakage; wrote the corresponding tests.
- **ACM3-24 - Data and model tests:** Implemented tests for data preprocessing, models, and model results.
- **ACM3-27 - SHAP explainability tests:** Tested the SHAP explainability implementation.
- **ACM3-28 - SHAP stability tests:** Tested the SHAP stability analysis across cross-validation folds.
- **ACM3-38 - Report QA:** Wrote the test subsection for the Data and Experimental Design section of the report.

## Nguyen Van Vi

**Role:** Pipeline / AI Engineer

Nguyen Van Vi was responsible for designing the end-to-end machine learning pipeline architecture, implementing reproducible dataset splitting and cross-validation protocols, leading the K-Fold validation and SHAP stability analysis epic, and authoring the corresponding methodology sections in the research report, with continuous technical alignment and support from Le Quang Thanh and Pham Minh Dang Tran.

### Completed work

- **ACM3-7 - Design ML pipeline workflow:** Designed the end-to-end experiment workflow and pipeline diagrams; revised and approved by Le Quang Thanh[cite: 1, 4].
- **ACM3-14 - Implement dataset splitting and stratified K-fold CV:** Built 80/20 splits and 5-fold CV generating persisted fold indices (`kfold_indices.csv`).
- **ACM3-16 - K-Fold validation & SHAP analysis:** Led Epic 4 coordinating cross-validation and SHAP stability execution; supervised by Le Quang Thanh and integrated model outputs from Pham Minh Dang Tran.
- **ACM3-26 - Implement SHAP stability analysis across CV folds:** Built the fold-ranking comparison module (Kendall's $\tau$, Spearman's $\rho$, Top-10 Jaccard) and heatmaps; supported by Pham Minh Dang Tran.
- **ACM3-31 - Write dataset splitting and CV methodology:** Authored the data partitioning and CV methodology sections of the report.
- **ACM3-33 - Write SHAP stability methodology:** Documented the mathematical metrics and stability methodology in the report.

## Team-Level Summary

The team completed the project through five broad work areas:

1. **Project initiation and alignment:** Team coordination, repository and Jira setup, research definition, and workflow planning.
2. **Data understanding and preparation:** Dataset provenance checks, exploratory analysis, preprocessing, imbalance handling, outlier handling, and feature reduction.
3. **Model development and comparison:** Selection and implementation of AdaBoost, XGBoost, and LightGBM, followed by metric-based comparison.
4. **Validation and explainability:** Stratified cross-validation, SHAP explainability, SHAP stability analysis, and QA testing.
5. **Reporting and delivery:** Research methodology, architecture, pipeline, presentation, final review, and project documentation.
