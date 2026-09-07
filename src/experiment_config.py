from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_EXPERIMENT_KEYS = {
    "random_state",
    "decision_threshold",
    "primary_metric",
    "balance_training",
    "output_dir",
    "cv",
}
SUPPORTED_PRIMARY_METRICS = {"roc_auc", "pr_auc", "recall", "f1"}
REQUIRED_PREPROCESSING_KEYS = {
    "test_size",
    "iqr_multiplier",
    "feature_variance_threshold",
    "feature_correlation_threshold",
}
DEFAULT_PREPROCESSING = {
    "test_size": 0.2,
    "iqr_multiplier": 1.5,
    "feature_variance_threshold": 1e-4,
    "feature_correlation_threshold": 0.9,
}


def load_experiment_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load and minimally validate the YAML experiment configuration.

    Returns the parsed configuration and the project root. Relative paths in
    ``config.yaml`` are resolved from the directory containing the config file,
    which makes runs reproducible regardless of the caller's working directory.
    """
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")

    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a YAML mapping at the top level.")

    for top_level_key in ("experiment", "datasets", "models"):
        if top_level_key not in config:
            raise ValueError(f"Missing top-level config section: '{top_level_key}'")

    experiment = config["experiment"]
    missing = REQUIRED_EXPERIMENT_KEYS.difference(experiment)
    if missing:
        raise ValueError(f"Missing experiment config keys: {sorted(missing)}")

    preprocessing = config.setdefault("preprocessing", dict(DEFAULT_PREPROCESSING))
    if not isinstance(preprocessing, dict):
        raise ValueError("Top-level 'preprocessing' configuration must be a mapping.")
    missing = REQUIRED_PREPROCESSING_KEYS.difference(preprocessing)
    if missing:
        raise ValueError(f"Missing preprocessing config keys: {sorted(missing)}")
    test_size = float(preprocessing["test_size"])
    if not 0.0 < test_size < 1.0:
        raise ValueError("preprocessing.test_size must be between 0 and 1.")

    cv = experiment["cv"]
    if int(cv.get("n_splits", 0)) < 2:
        raise ValueError("experiment.cv.n_splits must be at least 2.")

    threshold = float(experiment["decision_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("experiment.decision_threshold must be between 0 and 1.")

    primary_metric = str(experiment["primary_metric"])
    if primary_metric not in SUPPORTED_PRIMARY_METRICS:
        raise ValueError(
            f"Unsupported primary_metric '{primary_metric}'. "
            f"Choose one of {sorted(SUPPORTED_PRIMARY_METRICS)}."
        )

    if not config["datasets"]:
        raise ValueError("At least one dataset must be configured.")
    if not config["models"]:
        raise ValueError("At least one model must be configured.")

    for dataset_key, dataset in config["datasets"].items():
        required_dataset_keys = {"name", "train_path", "test_path", "target_column"}
        missing = required_dataset_keys.difference(dataset)
        if missing:
            raise ValueError(
                f"Dataset '{dataset_key}' is missing config keys: {sorted(missing)}"
            )

    shap = config.get("shap")
    if shap is not None:
        if not isinstance(shap, dict):
            raise ValueError("Top-level 'shap' configuration must be a mapping.")
        required_shap_keys = {"output_dir", "max_explain_samples", "background_samples", "top_k"}
        missing = required_shap_keys.difference(shap)
        if missing:
            raise ValueError(f"Missing SHAP config keys: {sorted(missing)}")

    return config, path.parent
