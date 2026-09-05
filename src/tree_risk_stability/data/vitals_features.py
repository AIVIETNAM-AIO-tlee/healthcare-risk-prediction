"""Vitals aggregation helpers stub — Phase 2."""

from __future__ import annotations

import pandas as pd

VITAL_COLUMNS = ("heart_rate", "systolic_bp", "diastolic_bp", "temperature", "spo2")


def aggregate_vitals(vitals: pd.DataFrame) -> pd.DataFrame:
    """Patient-level mean/std/min/max per vital column, indexed by patient_id."""
    raise NotImplementedError("Phase 2")
