---
phase: 6
title: "Analysis and Paper"
status: pending
priority: P2
effort: "4d"
dependencies: ["phase-05-statistical-evaluation.md"]
---

# Phase 6: Analysis and Paper

## Overview

Interpret results in analysis notebooks, then write the paper draft answering RQ1–RQ3 with the full evidence chain. No new package code except plotting polish; notebooks are narrative + calls into the existing package.

## Requirements

- Functional:
  - `notebooks/01-eda.ipynb` — per-dataset exploration: distributions, correlation snapshots, class balance, synthetic vitals patterns.
  - `notebooks/02-results-analysis.ipynb` — loads `results/tables/*.csv`, reproduces every paper figure, walks through RQ1/RQ2/RQ3 answers.
  - Paper draft `paper/draft.md` (Markdown, exportable to LaTeX later): IMRaD structure.
- Non-functional: every figure in draft traceable to a notebook cell output or table file; claims cite specific table/figure.

## Architecture

```
notebooks/{01-eda,02-results-analysis}.ipynb
paper/
├── draft.md            # main working document
└── figures/            # exported PNG/PDF copies used by draft
docs/paper-outline.md   # section-by-section claim → evidence map (written first)
```

## Related Code Files

- Create: both notebooks, `paper/draft.md`, `paper/figures/*`, `docs/paper-outline.md`
- Modify: `src/tree_risk_stability/visualization/plots.py` only if figure polish requires it

## Implementation Steps

1. Write `docs/paper-outline.md`: for each RQ — claim, supporting tables/figures, statistical tests cited, limitations. Review outline before drafting prose.
2. Build `01-eda.ipynb`: dataset catalog tables, distribution plots, missing-value summaries. Keep execution deterministic (fixed seeds where sampling).
3. Build `02-results-analysis.ipynb`: RQ1 leaderboard + CD diagrams; RQ2 stability ranking across models with variance tests; RQ3 W/Jaccard/CV comparisons plus concrete examples of top-k rank flips between folds.
4. Draft paper sections in order: Methods (from methodology doc), Results (from notebooks), Discussion (stability-vs-accuracy trade-off findings; sample-size effect on stability as key finding), Intro/Related Work (positioning vs IELF 2025, IEEE Access 2024, Alzheimer's 2026 from brainstorm report), Abstract last.
5. Reproducibility appendix: exact commands, environment freeze (`pip freeze > results/environment.txt`), seeds, runtime.
6. Final pass: verify every acceptance criterion from brainstorm report success metrics is demonstrably met.

## Success Criteria

- [ ] Both notebooks execute top-to-bottom without error on clean kernel
- [ ] Every figure/table referenced in draft exists in `paper/figures/`
- [ ] RQ1–RQ3 each have explicit answer paragraphs backed by statistics
- [ ] Limitations section covers: small-n multiclass noise, fold correlation, SHAP perturbation choice, dataset provenance (synthetic source)
- [ ] Full pipeline reruns from clone → install → download → run → aggregate → figures via documented commands

## Risk Assessment

- Narrative drift (results suggest different story than planned novelty claim): mitigation — outline-first workflow forces claim-evidence mapping before prose; update brainstorm-level framing if evidence demands.
- Notebook rot: mitigation — execute-all before commit; keep notebooks thin over the package.
