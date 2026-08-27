"""Pytest bootstrap: the src modules use bare imports (``config``, ``data.x``),
so make ``src/`` the import root before any test module imports them.

Shared fixtures live here too: pytest only auto-discovers fixtures from
conftest.py, not from helper modules imported by name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

NUMERIC = ["num1", "num2"]
CATEGORICAL = ["bin1", "multi1"]


@pytest.fixture()
def raw_df() -> pd.DataFrame:
	"""Binary-target frame mixing numeric + categorical + missing values.

	``num1`` has NaNs, ``bin1`` is a 2-category string column with a NaN,
	``multi1`` has 3 categories. Target is imbalanced (6/4) so stratification
	behavior is observable.
	"""
	return pd.DataFrame(
		{
			"target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
			"num1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
			"num2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
			"bin1": ["Y", "N", "N", "Y", None, "Y", "N", "N", "Y", "N"],
			"multi1": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"],
		}
	)


@pytest.fixture()
def numeric_columns() -> list[str]:
	return list(NUMERIC)


@pytest.fixture()
def categorical_columns() -> list[str]:
	return list(CATEGORICAL)

