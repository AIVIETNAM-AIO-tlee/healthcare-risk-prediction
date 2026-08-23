---
date: 2026-08-23
type: journal
tags: [research-planning, shap, cross-validation, healthcare-ml]
---

# Journal: Tree Risk Stability Study — Brainstorm & Planning

## Context

Fresh `Module3` workspace (empty repo). User researching "A Comparative Study of Tree-based Models for Healthcare Risk Prediction: Cross-validation and SHAP-based Stability Analysis" targeting a publishable paper. Dataset source: synthetic repo (n=50, Low/Med/High = 30/15/5).

## What Happened

1. Scout found empty repo + tiny synthetic dataset (50 patients). Flagged statistical fragility: High class = 5 samples.
2. Discovery rounds locked decisions: publishable paper goal; add 5 public datasets; native per-dataset targets; demographics+vitals aggregation features; repeated nested CV (5×10); Kendall's W + Spearman + Jaccard@k + SHAP-CV stability suite; Friedman/Nemenyi/Wilcoxon-Holm inference; modular package architecture; English docs; 2–4 week timeline.
3. Key challenge: user initially wanted multiclass labels forced onto binary datasets — argued against (label fabrication risk), resolved with unified protocol over native targets.
4. Literature scan confirmed novelty gap: no systematic dual-stability (performance + explanation) comparison of DT/RF/XGB across multiple healthcare datasets.
5. Brainstorm report written and approved; `/ck:plan --tdd` produced 6-phase plan (research → data layer → model pipeline → explainability → statistical evaluation → analysis/paper).

## Decisions

- Architecture B chosen: config-driven Python package, notebooks only for EDA/analysis (DRY, reproducible).
- TDD mode selected by user despite greenfield — metric code gets analytic-value tests.
- Multiclass SHAP aggregation pinned: mean of per-class mean|SHAP|; interventional perturbation with train-fold background.

## Reflection

The critical save was catching the multiclass-forcing idea early — desk-rejection risk removed cheaply at design time instead of after implementation. Small-n synthetic dataset reframed from bug to finding: sample-size effect on stability becomes a paper contribution.

## Next

Execute Phase 1 (dataset acquisition + methodology docs) via `/ck:cook` or start manually; full plan at `plans/260823-1936-tree-risk-study/`.
