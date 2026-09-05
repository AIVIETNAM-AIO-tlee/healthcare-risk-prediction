"""Nested CV runner stub — implemented in Phase 3."""

from __future__ import annotations


def make_spy_fit_context(train_idx, estimator_class):
    """QA seam: returns a context whose inner grid search fits `estimator_class`
    instances on outer-train rows only. The context must expose `.spy` (the last
    fitted estimator) and `run_inner_search(X, y)` performing >=2 inner fits."""
    raise NotImplementedError("Phase 3")


def build_inner_cv(y, config: dict):
    """Build the inner CV splitter for a dataset slug.

    Default: StratifiedKFold(3). When config["inner_cv_override"] is true
    (synthetic regime), the splitter must guarantee >=2 minority samples per
    INNER validation fold.
    """
    raise NotImplementedError("Phase 3")
