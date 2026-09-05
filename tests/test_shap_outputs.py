"""SHAP value generation and feature-importance validation (scope: SHAP
correctness for AdaBoost, XGBoost, LightGBM).

Validates the explainer contracts the study relies on:

- TreeExplainer works for XGBoost and LightGBM; SHAP values keep the input
  frame's (rows x features) layout and are finite.
- Local accuracy (additivity): base_value + sum(SHAP) must reconstruct the
  model's raw margin. LightGBM's ``predict(pred_raw=True)`` silently returns
  class labels, so the reference must come from ``booster_.predict(raw_score=True)``.
- AdaBoostClassifier is NOT supported by TreeExplainer (shap raises
  InvalidModelError); a PermutationExplainer on ``predict_proba`` is the
  fallback and satisfies local accuracy against the positive-class probability.
- Global importance = mean(|SHAP|) must rank the known signal feature first
  and broadly agree with each model's native ``feature_importances_``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import shap
import yaml
from pathlib import Path
from scipy.stats import spearmanr

SIGNAL_FEATURE = "f0"
WEAK_FEATURE = "f1"
FEATURES = [f"f{i}" for i in range(5)]
N_ROWS = 300
PERMUTATION_ROWS = 20


def _make_dataset() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(N_ROWS, 5)), columns=FEATURES)
    logits = 2.0 * X[SIGNAL_FEATURE] + 0.6 * X[WEAK_FEATURE] + rng.normal(0, 0.5, N_ROWS)
    y = (logits > 0).astype(int)
    return X, pd.Series(y, name="target")


@pytest.fixture(scope="module")
def dataset():
    return _make_dataset()


@pytest.fixture(scope="module")
def trained(dataset):
    X, y = dataset
    with (Path(__file__).resolve().parents[1] / "config.yaml").open("r", encoding="utf-8") as stream:
        models_config = yaml.safe_load(stream)["models"]
    from models.factory import build_model

    return {
        key: build_model(key, models_config[key], random_state=0).fit(X, y)
        for key in ("xgboost", "lightgbm", "adaboost")
    }


def global_importance(shap_values: np.ndarray) -> pd.Series:
    return pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES)


# ---------------------------------------------------------------------------
# Generation + dimensions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_key", ["xgboost", "lightgbm"])
def test_tree_explainer_values_match_input_dimensions(trained, dataset, model_key):
    X, _ = dataset

    values = shap.TreeExplainer(trained[model_key]).shap_values(X)

    assert isinstance(values, np.ndarray)
    assert values.shape == (len(X), len(FEATURES))
    assert np.isfinite(values).all()


def test_adaboost_rejected_by_tree_explainer(trained):
    from shap.utils._exceptions import InvalidModelError

    with pytest.raises(InvalidModelError):
        shap.TreeExplainer(trained["adaboost"])


@pytest.fixture(scope="module")
def adaboost_explanation(trained, dataset):
    X, _ = dataset
    masker = shap.maskers.Independent(X, max_samples=50)
    explainer = shap.Explainer(
        trained["adaboost"].predict_proba, masker, algorithm="permutation"
    )
    return explainer(X.iloc[:PERMUTATION_ROWS])


def test_adaboost_permutation_output_dimensions(adaboost_explanation):
    """predict_proba output -> (rows, features, classes) values/base/data."""
    exp = adaboost_explanation

    assert exp.values.shape == (PERMUTATION_ROWS, len(FEATURES), 2)
    assert exp.base_values.shape == (PERMUTATION_ROWS, 2)
    assert exp.data.shape == (PERMUTATION_ROWS, len(FEATURES))
    assert np.isfinite(exp.values).all()


# ---------------------------------------------------------------------------
# Local accuracy (the core correctness contract)
# ---------------------------------------------------------------------------
def test_xgboost_shap_reconstructs_raw_margin(trained, dataset):
    X, _ = dataset
    explainer = shap.TreeExplainer(trained["xgboost"])
    values = explainer.shap_values(X)
    margin = trained["xgboost"].predict(X, output_margin=True).astype(float)

    reconstructed = values.sum(axis=1) + float(explainer.expected_value)
    assert np.max(np.abs(reconstructed - margin)) < 1e-4


def test_lightgbm_shap_reconstructs_booster_raw_score(trained, dataset):
    X, _ = dataset
    explainer = shap.TreeExplainer(trained["lightgbm"])
    values = explainer.shap_values(X)
    # predict(pred_raw=True) returns class LABELS -- the raw logit reference
    # must come from the booster itself.
    raw_score = trained["lightgbm"].booster_.predict(X, raw_score=True)

    reconstructed = values.sum(axis=1) + float(explainer.expected_value)
    assert np.max(np.abs(reconstructed - raw_score)) < 1e-4


def test_adaboost_permutation_reconstructs_positive_probability(
    trained, dataset, adaboost_explanation
):
    X, _ = dataset
    exp = adaboost_explanation
    positive_probability = trained["adaboost"].predict_proba(X.iloc[:PERMUTATION_ROWS])[:, 1]

    reconstructed = exp.values[:, :, 1].sum(axis=1) + exp.base_values[:, 1]
    assert np.max(np.abs(reconstructed - positive_probability)) < 1e-5


# ---------------------------------------------------------------------------
# Feature correspondence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_key", ["xgboost", "lightgbm"])
def test_shap_columns_correspond_to_input_features(trained, dataset, model_key):
    """The synthetic target depends almost only on f0, so the f0 SHAP column
    must carry by far the largest attribution mass -- proving columns keep the
    input frame's feature order."""
    X, _ = dataset

    importance = global_importance(shap.TreeExplainer(trained[model_key]).shap_values(X))

    assert importance.idxmax() == SIGNAL_FEATURE
    assert importance[SIGNAL_FEATURE] > 2 * importance[WEAK_FEATURE]


# ---------------------------------------------------------------------------
# Global importance + rankings
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_key", ["xgboost", "lightgbm"])
def test_global_importance_ranking_matches_signal(trained, dataset, model_key):
    X, _ = dataset

    ranking = global_importance(shap.TreeExplainer(trained[model_key]).shap_values(X)).sort_values(
        ascending=False
    )

    assert ranking.index[0] == SIGNAL_FEATURE
    assert set(ranking.index[:2]) == {SIGNAL_FEATURE, WEAK_FEATURE}


def test_adaboost_global_importance_ranking_matches_signal(
    dataset, adaboost_explanation
):
    ranking = global_importance(adaboost_explanation.values[:, :, 1]).sort_values(
        ascending=False
    )

    assert ranking.index[0] == SIGNAL_FEATURE


@pytest.mark.parametrize("model_key", ["xgboost", "lightgbm", "adaboost"])
def test_shap_ranking_agrees_with_native_feature_importances(
    trained, dataset, adaboost_explanation, model_key
):
    X, _ = dataset
    model = trained[model_key]
    if model_key == "adaboost":
        shap_ranking = global_importance(adaboost_explanation.values[:, :, 1])
    else:
        shap_ranking = global_importance(shap.TreeExplainer(model).shap_values(X))

    native = pd.Series(model.feature_importances_, index=FEATURES)
    correlation, _ = spearmanr(
        shap_ranking[FEATURES].to_numpy(), native[FEATURES].to_numpy()
    )

    assert correlation >= 0.6, f"SHAP vs native importance rank mismatch: {correlation=}"


# ---------------------------------------------------------------------------
# Determinism + explanation object / visualization output
# ---------------------------------------------------------------------------
def test_tree_explainer_is_deterministic(trained, dataset):
    X, _ = dataset
    explainer = shap.TreeExplainer(trained["xgboost"])

    first = explainer.shap_values(X)
    second = explainer.shap_values(X)

    np.testing.assert_array_equal(first, second)


def test_summary_plot_produces_figure(trained, dataset):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, _ = dataset
    values = shap.TreeExplainer(trained["xgboost"]).shap_values(X)

    shap.summary_plot(values, X, show=False)

    figure = plt.gcf()
    assert figure.axes, "summary plot produced no axes"
    plt.close("all")


def test_explanation_object_contract(trained, dataset):
    X, _ = dataset
    explainer = shap.TreeExplainer(trained["xgboost"])
    explanation = explainer(X)

    assert explanation.values.shape == (len(X), len(FEATURES))
    assert np.ndim(explanation.base_values) <= 1
    assert explanation.data.shape == (len(X), len(FEATURES))
