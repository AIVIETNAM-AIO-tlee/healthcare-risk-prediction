# Contributing

Thank you for contributing to the Healthcare Risk Prediction project. This repository contains a research pipeline, so contributions should keep the code, generated results, report claims, and experimental configuration consistent with one another.

## Before You Start

1. Create or identify the related Jira task before making a substantial change.
2. Read the task scope and Definition of Done.
3. Check the current branch and worktree so existing changes are not overwritten.
4. Do not commit raw data or generated files unless the task explicitly requires them.

Changes involving datasets, preprocessing, models, metrics, experiment design, SHAP analysis, or report conclusions should be linked to the relevant Jira task.

## Development Workflow

Use a focused branch for each task. Keep changes small enough to review and use a descriptive branch name, for example:

```text
feature/fold-local-preprocessing
fix/roc-curve-output
report/update-results-discussion
```

From the repository root, install the project dependencies before running the pipeline:

```powershell
python -m pip install -r requirements.txt
git lfs install
git lfs pull
```

Run the reproducible pipeline in this order when data or results need to be regenerated:

```powershell
python -m src.main
python -m src.experiments.run_models --config config.yaml
python -m src.experiments.run_shap --config config.yaml
```

Use notebooks for EDA, visualization, and result analysis. Do not make a notebook-only change to the reproducible preprocessing or experiment pipeline without updating the corresponding source module.

## Research and Data Standards

- Use `config.yaml` for experiment-level settings such as seeds, splits, metrics, model parameters, and SHAP settings.
- Keep dataset metadata and dataset-specific cleaning rules in `src/config.py`.
- Validate configuration changes through `src/experiment_config.py`.
- Keep the hold-out test partition outside model selection and cross-validation.
- Fit learned preprocessing operations only on the relevant training rows.
- Reuse the same persisted fold manifest for model evaluation and SHAP analysis.
- Do not change a metric, split rule, seed, or model configuration without documenting the reason and updating the report where necessary.
- Treat SHAP importance as model explanation, not causal evidence.
- Check dataset provenance, licensing, and redistribution conditions before adding or sharing data.
- Never include private, identifying, or restricted healthcare data in the repository.

When adding a dataset or changing preprocessing, document its source, target column, feature handling, missing-value policy, class balance, and expected output files.

## Code Style

- Follow the existing Python structure and naming conventions.
- Keep public functions focused and reuse existing helpers before adding new abstractions.
- Prefer clear, typed interfaces and deterministic behavior.
- Avoid unrelated refactors in a task-specific pull request.
- Add comments only when they clarify non-obvious behavior, especially leakage controls or data-provenance constraints.
- Keep generated outputs separate from source code and update documentation when output paths change.

## Testing and Validation

Run at least the checks relevant to the changed files:

```powershell
python -m compileall -q src tests
python -m pytest -q
```

For report changes, build from the report directory with the available LaTeX toolchain and check that:

- the report compiles without fatal errors;
- all referenced figures and tables exist;
- table and figure labels are unique;
- result values match the current CSV artifacts;
- citations used in the text exist in `references.bib`.

For experiment changes, verify that the affected output files are regenerated and inspect the resulting CSVs rather than relying only on console output.

## Pull Requests

A pull request should include:

- a link to the Jira task;
- a short description of the problem and change;
- the validation commands that were run;
- any regenerated result or report artifacts;
- known limitations or follow-up work.

Reviewers should be able to trace the change from the Jira task to the implementation, test or experiment output, and report claim where applicable.

Use this checklist before requesting review:

- [ ] The change is limited to the Jira task scope.
- [ ] Configuration and documentation match the implementation.
- [ ] Tests or focused validation were run.
- [ ] No data leakage or hold-out contamination was introduced.
- [ ] Result tables and figures were regenerated when needed.
- [ ] Report conclusions do not overstate the evidence.
- [ ] Dataset sources and citations are preserved.
- [ ] Temporary files, caches, credentials, and local environment files are excluded.

## Review Responsibilities

Authors are responsible for explaining the change and reporting validation results. Reviewers should check correctness, reproducibility, data leakage, metric validity, dataset provenance, and consistency between code, artifacts, and the report. Tech Lead or QA review should be requested for changes that affect shared interfaces, experiment outputs, evaluation methodology, or final report claims.

## Questions

For questions about scope, ownership, or Definition of Done, use the related Jira task and involve the responsible team role before expanding the implementation beyond the agreed scope.
