"""Helpers for the tree-risk-stability CV leakage tests (study-owned).

Plain module (deliberately NOT named conftest.py) so pytest's bare-name
conftest import used by the root test suite can never resolve to this file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def make_multiclass_mini(n_per_class: dict = None) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic-regime mini bundle: Low/Medium/High with tiny High class."""
    if n_per_class is None:
        n_per_class = {"Low": 6, "Medium": 3, "High": 1}
    rows, labels = [], []
    for label, k in n_per_class.items():
        for _ in range(k):
            rows.append([RNG.normal(60, 10), RNG.normal(110, 15), RNG.normal(25, 3)])
            labels.append(label)
    X = pd.DataFrame(rows, columns=["age", "sys_bp", "bmi"])
    y = pd.Series(labels, name="risk_level", dtype="category")
    y = y.cat.reorder_categories(["Low", "Medium", "High"])
    return X, y
