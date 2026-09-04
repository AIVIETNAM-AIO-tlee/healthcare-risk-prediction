"""Unit tests for SHAP stability module."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau, spearmanr

from src.evaluation.shap_stability import (
    compute_fold_shap_importance,
    summarize_shap_stability,
)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _make_fold_importance(
    features: list[str],
    n_folds: int = 5,
    *,
    identical: bool = False,
    dataset_key: str = "ds1",
    dataset_name: str = "Dataset 1",
    model_key: str = "model1",
    model_name: str = "Model 1",
) -> pd.DataFrame:
    """Build a synthetic shap_fold_importance DataFrame.

    If identical=True, every fold gets exactly the same SHAP values
    so rankings should be perfectly correlated (tau=1, rho=1, jaccard=1).
    """
    rng = np.random.RandomState(42)
    rows: list[dict[str, Any]] = []
    base_importance = rng.rand(len(features))
    for fold in range(1, n_folds + 1):
        importance = base_importance if identical else rng.rand(len(features))
        for feat, imp in zip(features, importance):
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset_name": dataset_name,
                    "model_key": model_key,
                    "model_name": model_name,
                    "fold": fold,
                    "feature": feat,
                    "mean_abs_shap": float(imp),
                }
            )
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# Tests for summarize_shap_stability
# ──────────────────────────────────────────────────────────────
class TestSummarizeShapStability:
    """Tests for summarize_shap_stability."""

    def test_perfect_stability(self):
        """Identical rankings across folds → tau=1, rho=1, jaccard=1."""
        features = [f"f{i}" for i in range(20)]
        df = _make_fold_importance(features, n_folds=5, identical=True)
        result = summarize_shap_stability(df, top_k=10)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["n_folds"] == 5
        assert row["n_features"] == 20
        assert row["top_k"] == 10
        np.testing.assert_allclose(row["kendall_tau_mean"], 1.0, atol=1e-10)
        np.testing.assert_allclose(row["spearman_rho_mean"], 1.0, atol=1e-10)
        np.testing.assert_allclose(row["top_k_jaccard_mean"], 1.0, atol=1e-10)

    def test_random_rankings_below_perfect(self):
        """Random rankings should produce correlation < 1."""
        features = [f"f{i}" for i in range(20)]
        df = _make_fold_importance(features, n_folds=5, identical=False)
        result = summarize_shap_stability(df, top_k=10)

        row = result.iloc[0]
        assert row["kendall_tau_mean"] < 1.0
        assert row["spearman_rho_mean"] < 1.0

    def test_top_k_larger_than_features(self):
        """top_k > n_features should be clamped."""
        features = [f"f{i}" for i in range(5)]
        df = _make_fold_importance(features, n_folds=3, identical=True)
        result = summarize_shap_stability(df, top_k=100)

        row = result.iloc[0]
        assert row["top_k"] == 5  # clamped to n_features
        np.testing.assert_allclose(row["top_k_jaccard_mean"], 1.0, atol=1e-10)

    def test_top_k_validation(self):
        """top_k < 1 should raise ValueError."""
        features = ["f0", "f1"]
        df = _make_fold_importance(features)
        with pytest.raises(ValueError, match="top_k must be at least 1"):
            summarize_shap_stability(df, top_k=0)

    def test_missing_columns(self):
        """Missing required columns should raise ValueError."""
        df = pd.DataFrame({"wrong_column": [1]})
        with pytest.raises(ValueError, match="missing required columns"):
            summarize_shap_stability(df)

    def test_multiple_groups(self):
        """Multiple dataset/model combos should produce multiple rows."""
        features = [f"f{i}" for i in range(10)]
        df1 = _make_fold_importance(features, n_folds=3, dataset_key="ds1", model_key="m1")
        df2 = _make_fold_importance(
            features, n_folds=3, dataset_key="ds2", model_key="m1", dataset_name="Dataset 2"
        )
        df3 = _make_fold_importance(
            features, n_folds=3, dataset_key="ds1", model_key="m2", model_name="Model 2"
        )
        combined = pd.concat([df1, df2, df3], ignore_index=True)
        result = summarize_shap_stability(combined, top_k=5)
        assert len(result) == 3

    def test_two_folds_produces_single_pair(self):
        """With 2 folds there's exactly 1 pair; verify n_folds=2."""
        features = [f"f{i}" for i in range(10)]
        df = _make_fold_importance(features, n_folds=2)
        result = summarize_shap_stability(df, top_k=5)
        assert result.iloc[0]["n_folds"] == 2

    def test_single_fold_no_crash(self):
        """With only 1 fold, there are 0 pairs → should not crash, should report 1.0."""
        features = [f"f{i}" for i in range(10)]
        df = _make_fold_importance(features, n_folds=1)
        result = summarize_shap_stability(df, top_k=5)
        row = result.iloc[0]
        assert row["n_folds"] == 1
        np.testing.assert_allclose(row["kendall_tau_mean"], 1.0, atol=1e-10)
        np.testing.assert_allclose(row["spearman_rho_mean"], 1.0, atol=1e-10)
        np.testing.assert_allclose(row["top_k_jaccard_mean"], 1.0, atol=1e-10)

    def test_single_feature_no_nan(self):
        """With 1 feature, rankings are trivially [1]; should report 1.0, not 0.0 or NaN."""
        df = _make_fold_importance(["only_feature"], n_folds=3)
        result = summarize_shap_stability(df, top_k=1)
        row = result.iloc[0]
        np.testing.assert_allclose(row["kendall_tau_mean"], 1.0, atol=1e-10)
        np.testing.assert_allclose(row["spearman_rho_mean"], 1.0, atol=1e-10)
        np.testing.assert_allclose(row["top_k_jaccard_mean"], 1.0, atol=1e-10)

    def test_two_features_identical(self):
        """Two features, identical across folds → perfect stability."""
        df = _make_fold_importance(["f0", "f1"], n_folds=3, identical=True)
        result = summarize_shap_stability(df, top_k=1)
        row = result.iloc[0]
        np.testing.assert_allclose(row["kendall_tau_mean"], 1.0, atol=1e-10)
        np.testing.assert_allclose(row["spearman_rho_mean"], 1.0, atol=1e-10)

    def test_manual_kendall_spearman_values(self):
        """Manually verify the Kendall/Spearman computation for a 2-fold case."""
        features = [f"f{i}" for i in range(5)]
        fold1_imp = [0.5, 0.3, 0.1, 0.4, 0.2]
        fold2_imp = [0.5, 0.2, 0.3, 0.4, 0.1]
        rows: list[dict[str, Any]] = []
        for fold_idx, importances in enumerate([fold1_imp, fold2_imp], 1):
            for feat, imp in zip(features, importances):
                rows.append(
                    {
                        "dataset_key": "ds",
                        "dataset_name": "DS",
                        "model_key": "m",
                        "model_name": "M",
                        "fold": fold_idx,
                        "feature": feat,
                        "mean_abs_shap": imp,
                    }
                )
        df = pd.DataFrame(rows)
        result = summarize_shap_stability(df, top_k=3)

        # Compute expected values manually
        pivot = df.pivot_table(index="fold", columns="feature", values="mean_abs_shap")
        ranks = pivot.rank(axis=1, ascending=False, method="average")
        r1 = ranks.iloc[0]
        r2 = ranks.iloc[1]
        expected_kendall = kendalltau(r1, r2).statistic
        expected_spearman = spearmanr(r1, r2).statistic

        row = result.iloc[0]
        np.testing.assert_allclose(row["kendall_tau_mean"], expected_kendall, atol=1e-10)
        np.testing.assert_allclose(row["spearman_rho_mean"], expected_spearman, atol=1e-10)

    def test_output_columns(self):
        """Verify all expected output columns are present."""
        features = [f"f{i}" for i in range(10)]
        df = _make_fold_importance(features, n_folds=3)
        result = summarize_shap_stability(df, top_k=5)
        expected_cols = {
            "dataset_key",
            "dataset_name",
            "model_key",
            "model_name",
            "n_folds",
            "n_features",
            "top_k",
            "kendall_tau_mean",
            "spearman_rho_mean",
            "top_k_jaccard_mean",
        }
        assert set(result.columns) == expected_cols


# ──────────────────────────────────────────────────────────────
# Tests for compute_fold_shap_importance
# ──────────────────────────────────────────────────────────────
class TestComputeFoldShapImportance:
    """Tests for compute_fold_shap_importance with real sklearn models."""

    @staticmethod
    def _make_data():
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series((rng.rand(100) > 0.5).astype(int))
        return X, y

    def test_with_xgboost(self):
        """XGBoost should produce valid SHAP importances via TreeExplainer."""
        from xgboost import XGBClassifier

        X, y = self._make_data()
        model = XGBClassifier(n_estimators=10, random_state=42, verbosity=0)
        model.fit(X, y)

        result = compute_fold_shap_importance(model, X)
        assert isinstance(result, pd.Series)
        assert result.name == "mean_abs_shap"
        assert len(result) == 5
        assert list(result.index) == [f"f{i}" for i in range(5)]
        assert (result >= 0).all()

    def test_with_lightgbm(self):
        """LightGBM should produce valid SHAP importances."""
        from lightgbm import LGBMClassifier

        X, y = self._make_data()
        model = LGBMClassifier(n_estimators=10, random_state=42, verbosity=-1)
        model.fit(X, y)

        result = compute_fold_shap_importance(model, X)
        assert isinstance(result, pd.Series)
        assert len(result) == 5
        assert (result >= 0).all()

    def test_with_adaboost(self):
        """AdaBoost should produce valid SHAP importances via optimized tree SHAP."""
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier

        X, y = self._make_data()
        model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=10,
            random_state=42,
        )
        model.fit(X, y)

        result = compute_fold_shap_importance(model, X)
        assert isinstance(result, pd.Series)
        assert result.name == "mean_abs_shap"
        assert len(result) == 5
        assert list(result.index) == [f"f{i}" for i in range(5)]
        assert (result >= 0).all()

    def test_with_adaboost_depth2(self):
        """AdaBoost with deeper trees (depth > 1) should also succeed via tree loop."""
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier

        X, y = self._make_data()
        model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2),
            n_estimators=5,
            random_state=42,
        )
        model.fit(X, y)

        result = compute_fold_shap_importance(model, X)
        assert isinstance(result, pd.Series)
        assert len(result) == 5
        assert (result >= 0).all()

    def test_empty_fold_importance(self):
        """An empty DataFrame with required columns returns empty result with correct schema."""
        empty_df = pd.DataFrame(
            columns=[
                "dataset_key",
                "dataset_name",
                "model_key",
                "model_name",
                "fold",
                "feature",
                "mean_abs_shap",
            ]
        )
        result = summarize_shap_stability(empty_df, top_k=5)
        assert len(result) == 0
        assert "kendall_tau_mean" in result.columns
        assert "top_k_jaccard_mean" in result.columns

    def test_tied_rankings_identical_across_folds(self):
        """When all features have equal SHAP in every fold, agreement is 1.0."""
        features = [f"f{i}" for i in range(5)]
        rows = []
        for fold in (1, 2):
            for feat in features:
                rows.append(
                    {
                        "dataset_key": "ds",
                        "dataset_name": "DS",
                        "model_key": "m",
                        "model_name": "M",
                        "fold": fold,
                        "feature": feat,
                        "mean_abs_shap": 0.5,  # all tied
                    }
                )
        df = pd.DataFrame(rows)
        result = summarize_shap_stability(df, top_k=3)
        row = result.iloc[0]
        np.testing.assert_allclose(row["kendall_tau_mean"], 1.0, atol=1e-10)
        np.testing.assert_allclose(row["spearman_rho_mean"], 1.0, atol=1e-10)
        np.testing.assert_allclose(row["top_k_jaccard_mean"], 1.0, atol=1e-10)

