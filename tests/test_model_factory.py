from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

from src.models.factory import build_model


def test_builds_all_three_model_types() -> None:
    configs = {
        "adaboost": {"params": {"estimator": {"max_depth": 1}, "n_estimators": 5}},
        "xgboost": {"params": {"n_estimators": 5, "verbosity": 0}},
        "lightgbm": {"params": {"n_estimators": 5, "verbosity": -1}},
    }

    assert isinstance(build_model("adaboost", configs["adaboost"], 42), AdaBoostClassifier)
    assert isinstance(build_model("xgboost", configs["xgboost"], 42), XGBClassifier)
    assert isinstance(build_model("lightgbm", configs["lightgbm"], 42), LGBMClassifier)
