"""Data layer stubs — implemented in Phase 2 (plans/260823-1936-tree-risk-study/phase-02-data-layer.md)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DatasetBundle:
    """Uniform dataset container per phase-02 Requirements."""

    X: pd.DataFrame
    y: pd.Series
    task_type: str  # "binary" | "multiclass"
    feature_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def load_dataset(slug: str) -> DatasetBundle:
    """Load a dataset bundle by slug. Registry of 6 slugs defined in configs/experiments/."""
    raise NotImplementedError("Phase 2")
