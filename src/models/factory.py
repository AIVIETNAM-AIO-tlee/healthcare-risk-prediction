from __future__ import annotations

from copy import deepcopy
from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def _import_xgboost():
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "XGBoost is required for the 'xgboost' model. Install requirements.txt first."
        ) from exc
    return XGBClassifier


def _import_lightgbm():
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "LightGBM is required for the 'lightgbm' model. Install requirements.txt first."
        ) from exc
    return LGBMClassifier


def build_model(model_key: str, model_config: dict[str, Any], random_state: int) -> ClassifierMixin:
    """Construct one fresh classifier from ``config.yaml``.

    A new estimator must be created for every CV fold so no fitted state leaks
    from one fold into another. Random seeds are injected centrally to guarantee
    repeatable experiments.
    """
    key = model_key.strip().lower()
    params = deepcopy(model_config.get("params", {}))

    if key == "adaboost":
        estimator_params = params.pop("estimator", {})
        base_estimator = DecisionTreeClassifier(
            random_state=random_state,
            **estimator_params,
        )
        return AdaBoostClassifier(
            estimator=base_estimator,
            random_state=random_state,
            **params,
        )

    if key == "xgboost":
        XGBClassifier = _import_xgboost()
        params.setdefault("random_state", random_state)
        return XGBClassifier(**params)

    if key == "lightgbm":
        LGBMClassifier = _import_lightgbm()
        params.setdefault("random_state", random_state)
        return LGBMClassifier(**params)

    raise ValueError(
        f"Unknown model key '{model_key}'. Supported models: adaboost, xgboost, lightgbm."
    )
