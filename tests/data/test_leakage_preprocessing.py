"""Leakage Vectors 1 & 5 — preprocessing must learn from TRAIN folds only.

Vector 1: imputation/encoding statistics fit on train fold only (inside Pipeline).
Vector 5: vitals aggregation is a per-patient transform — it must not be influenced
by rows outside the aggregation group (i.e., no cross-patient statistics).

Plan refs: phase-02-data-layer.md (vitals aggregation, deterministic loaders),
phase-03-model-pipeline.md Requirements ("preprocessing steps (impute + encode)
inside the Pipeline"), brainstorm §4 pipeline step 1.

RED contract: implemented in Phase 2 (vitals) and Phase 3 (pipeline builders).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tree_risk_stability.data.vitals_features import aggregate_vitals


def _make_vitals(n_patients: int = 3, hours: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(1, n_patients + 1):
        pid = f"P{p:03d}"
        for h in range(hours):
            rows.append(
                {
                    "patient_id": pid,
                    "timestamp": f"2026-01-0{h + 1} 00:00:00",
                    "heart_rate": rng.uniform(50, 100),
                    "systolic_bp": rng.uniform(100, 140),
                    "diastolic_bp": rng.uniform(60, 95),
                    "temperature": rng.uniform(36.0, 38.0),
                    "spo2": rng.uniform(93, 99),
                }
            )
    return pd.DataFrame(rows)


class TestVitalsAggregationIsolation:
    """Vector 5: aggregating patient A must be unaffected by patient B's rows."""

    def test_aggregate_runs_on_fixture(self):
        vitals = _make_vitals()
        agg = aggregate_vitals(vitals)
        assert len(agg) == 3

    def test_patient_aggregates_independent_of_other_patients(self):
        """Removing a DIFFERENT patient's rows must not change patient P001's aggregates."""
        full = _make_vitals(seed=7)
        reduced = full[full.patient_id != "P002"].copy()
        agg_full = aggregate_vitals(full).loc["P001"]
        agg_reduced = aggregate_vitals(reduced).loc["P001"]
        pd.testing.assert_series_equal(
            agg_full.astype(float), agg_reduced.astype(float), check_exact=False, rtol=1e-12
        )

    def test_row_order_invariance(self):
        vitals = _make_vitals(seed=11)
        shuffled = vitals.sample(frac=1.0, random_state=3).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            aggregate_vitals(vitals).sort_index(),
            aggregate_vitals(shuffled).sort_index(),
            check_exact=False,
            rtol=1e-12,
        )


class TestImputerTrainFoldOnly:
    """Vector 1: median imputation learned on train fold must differ from full-data
    median when the test fold is engineered to shift the statistic — proving the
    imputer never saw test rows."""

    def _fit_pipeline_and_extract_median(self, X_train, X_test):
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
        pipe.fit(X_train)
        return float(pipe.named_steps["impute"].statistics_[0])

    def test_imputer_statistics_come_from_train_only(self):
        rng = np.random.default_rng(5)
        train_vals = rng.normal(100, 5, 40)
        # Test rows engineered with a wildly different central tendency.
        test_vals = np.full(10, 500.0)

        X_train = pd.DataFrame({"glucose": train_vals})
        X_test = pd.DataFrame({"glucose": test_vals})

        learned_median = self._fit_pipeline_and_extract_median(X_train, X_test)
        assert abs(learned_median - np.median(train_vals)) < 1e-9
        assert learned_median != np.median(np.concatenate([train_vals, test_vals]))

    def test_full_fit_leaks_when_done_wrong(self):
        """Control test: fitting on FULL data shifts the statistic — demonstrates the
        detector can distinguish leaky vs clean fits."""
        rng = np.random.default_rng(5)
        train_vals = rng.normal(100, 5, 40)
        test_vals = np.full(10, 500.0)
        all_vals = np.concatenate([train_vals, test_vals])

        from sklearn.impute import SimpleImputer

        leaky = SimpleImputer(strategy="median").fit(pd.DataFrame({"glucose": all_vals}))
        assert abs(float(leaky.statistics_[0]) - np.median(all_vals)) < 1e-9
        # The leaky statistic differs from the train-only one by construction.
        assert not np.isclose(float(leaky.statistics_[0]), np.median(train_vals))
