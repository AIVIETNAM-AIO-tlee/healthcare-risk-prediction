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

    return config, path.parent
