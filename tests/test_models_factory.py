"""Model factory tests (scope: 'Test AdaBoost, XGBoost, and LightGBM training
and prediction').

Every CV fold must get a fresh, seed-injected estimator, so these tests pin
construction (config params + random_state), isolation between builds, and a
full train/predict round trip with valid output shapes and values for all
three model keys from config.yaml.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from models.factory import build_model


@pytest.fixture(scope="module")
def config_models() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)["models"]


def _mini_dataset(n: int = 240, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % 4 == 0).astype(int)  # 25% positives, both classes solid
    X = pd.DataFrame(rng.normal(size=(n, 4)) + y[:, None] * 0.8,
                     columns=["f0", "f1", "f2", "f3"])
    return X, pd.Series(y, name="target")


@pytest.mark.parametrize("model_key", ["adaboost", "xgboost", "lightgbm"])
def test_build_returns_expected_estimator_types(config_models, model_key):
    model = build_model(model_key, config_models[model_key], random_state=42)

    expected = {"adaboost": AdaBoostClassifier, "xgboost": "XGBClassifier",
                "lightgbm": "LGBMClassifier"}[model_key]
    if isinstance(expected, str):
        assert type(model).__name__ == expected
    else:
        assert isinstance(model, expected)


def test_adaboost_uses_configured_stump_and_seed(config_models):
    model = build_model("adaboost", config_models["adaboost"], random_state=42)

    assert isinstance(model, AdaBoostClassifier)
    assert isinstance(model.estimator, DecisionTreeClassifier)
    assert model.estimator.max_depth == config_models["adaboost"]["params"]["estimator"]["max_depth"]
    assert model.random_state == 42


@pytest.mark.parametrize("model_key", ["adaboost", "xgboost", "lightgbm"])
def test_random_state_injected_central(config_models, model_key):
    model = build_model(model_key, config_models[model_key], random_state=7)

    assert getattr(model, "random_state", None) == 7


def test_unknown_model_key_raises():
    with pytest.raises(ValueError, match="Unknown model key"):
        build_model("knn", {}, random_state=0)


def test_model_key_matching_is_case_and_space_insensitive(config_models):
    model = build_model("  AdaBoost ", config_models["adaboost"], random_state=1)

    assert isinstance(model, AdaBoostClassifier)


def test_config_params_are_not_mutated_by_build(config_models):
    config = {"params": {"n_estimators": 9}}

    build_model("adaboost", config, random_state=0)

    assert config["params"] == {"n_estimators": 9}


@pytest.mark.parametrize("model_key", ["adaboost", "xgboost", "lightgbm"])
def test_train_predict_round_trip_valid_outputs(config_models, model_key):
    X, y = _mini_dataset()
    model = build_model(model_key, config_models[model_key], random_state=42)

    model.fit(X, y)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    assert predictions.shape == (len(X),)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert probabilities.shape == (len(X), 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6)


@pytest.mark.parametrize("model_key", ["adaboost", "xgboost", "lightgbm"])
def test_same_seed_produces_identical_predictions(config_models, model_key):
    """Fold isolation contract: a fresh build with the same seed must
    reproduce the same fit, so no fitted state can silently leak between
    folds through the factory."""
    X, y = _mini_dataset()

    first = build_model(model_key, config_models[model_key], random_state=42).fit(X, y)
    second = build_model(model_key, config_models[model_key], random_state=42).fit(X, y)

    np.testing.assert_array_equal(first.predict(X), second.predict(X))
