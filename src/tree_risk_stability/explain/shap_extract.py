"""SHAP feature-importance extraction (study Phase 4 surface).

`extract_ranking` is the QA-mandated seam: `background` is an explicit
argument so callers can be audited for train-fold-only provenance. When a
background is supplied the interventional TreeExplainer path is engaged and
the background actively shapes the attributions; without one the classical
tree-path-dependent path is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def _positive_class_values(values) -> np.ndarray:
    """Normalize shap output shapes to (n_rows, n_features).

    Recent shap returns (n, features) for binary margin outputs but
    (n, features, classes) for some sklearn estimators; older versions return
    a list of per-class arrays. The positive class is always the last one.
    """
    array = np.asarray(values)
    if isinstance(values, list):
        array = np.asarray(values[-1])
    if array.ndim == 3:
        array = array[:, :, -1]
    if array.ndim != 2:
        raise ValueError(f"Unexpected SHAP values shape: {array.shape}")
    return array


def extract_ranking(model, X_fold: pd.DataFrame, background=None) -> pd.Series:
    """Feature -> mean|SHAP| sorted desc. Interventional TreeExplainer;
    background must come from the TRAIN fold only."""
    if background is None:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.TreeExplainer(
            model, data=background, feature_perturbation="interventional"
        )
    values = _positive_class_values(explainer.shap_values(X_fold))
    if values.shape[1] != len(X_fold.columns):
        raise ValueError(
            f"SHAP column count {values.shape[1]} does not match features "
            f"{len(X_fold.columns)}"
        )
    importance = np.abs(values).mean(axis=0)
    return pd.Series(importance, index=X_fold.columns).sort_values(ascending=False)
