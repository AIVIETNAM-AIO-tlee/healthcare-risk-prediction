"""QA tests for PR #8 (SHAP explainability + RQ3 stability analysis).

Targets the PR's new code without modifying it:
- src/explainability/shap_analysis.py (sampling, shape normalization,
  fold-importance extraction, pairwise stability, summaries)
- src/experiments/run_shap.py (end-to-end orchestration + output contract)
- the shipped result artifacts under results/shap_explainability/

Every test self-skips when the PR code is absent (e.g. on main before merge),
so the suite stays green regardless of merge state.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _shap_analysis():
    try:
        from explainability import shap_analysis
    except ImportError:
        pytest.skip("PR #8 src/explainability not present on this branch")
    return shap_analysis


def _run_shap():
    try:
        from experiments import run_shap
    except ImportError:
        pytest.skip("PR #8 src/experiments/run_shap.py not present on this branch")
    return run_shap


# ---------------------------------------------------------------------------
# _sample_rows
# ---------------------------------------------------------------------------
def test_sample_rows_deterministic_and_sorted():
    sa = _shap_analysis()
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(50, 2)), columns=["a", "b"])

    first = sa._sample_rows(frame, max_samples=10, random_state=7)
    second = sa._sample_rows(frame, max_samples=10, random_state=7)

    assert list(first.columns) == ["a", "b"]
    assert first.index.is_monotonic_increasing  # sorted by index
    pd.testing.assert_frame_equal(first, second)


def test_sample_rows_returns_copy_when_small_enough():
    sa = _shap_analysis()
    frame = pd.DataFrame({"a": [1.0, 2.0]})

    sampled = sa._sample_rows(frame, max_samples=10, random_state=0)

    assert len(sampled) == 2
    assert sampled is not frame


def test_sample_rows_rejects_non_positive_max_samples():
    sa = _shap_analysis()

    with pytest.raises(ValueError, match="max_samples must be positive"):
        sa._sample_rows(pd.DataFrame({"a": [1.0]}), max_samples=0, random_state=0)


# ---------------------------------------------------------------------------
# _positive_class_values (shape normalization)
# ---------------------------------------------------------------------------
def test_positive_class_values_handles_all_documented_shapes():
    sa = _shap_analysis()

    two_dim = np.arange(6.0).reshape(2, 3)
    assert sa._positive_class_values(two_dim, n_features=3).shape == (2, 3)

    three_dim = np.arange(12.0).reshape(2, 3, 2)
    np.testing.assert_array_equal(sa._positive_class_values(three_dim, 3), three_dim[:, :, 1])

    transposed = np.arange(12.0).reshape(2, 2, 3)
    np.testing.assert_array_equal(sa._positive_class_values(transposed, 3), transposed[:, 1, :])

    legacy_list = [np.zeros((2, 3)), np.ones((2, 3))]
    np.testing.assert_array_equal(sa._positive_class_values(legacy_list, 3), np.ones((2, 3)))


def test_positive_class_values_rejects_bad_shapes():
    sa = _shap_analysis()

    with pytest.raises(ValueError, match="two classes"):
        sa._positive_class_values([np.zeros((2, 3))], n_features=3)
    with pytest.raises(ValueError, match="Unexpected SHAP shape"):
        sa._positive_class_values(np.zeros((2, 4)), n_features=3)
    with pytest.raises(ValueError, match="Unsupported SHAP value shape"):
        sa._positive_class_values(np.zeros(6), n_features=3)


# ---------------------------------------------------------------------------
# compute_fold_shap_importance (real models, tiny data)
# ---------------------------------------------------------------------------
def _tiny_dataset(n: int = 120):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    y = pd.Series((2.0 * X.f0 + rng.normal(0, 0.5, n) > 0).astype(int))
    return X, y


@pytest.fixture(scope="module")
def tiny_models():
    from sklearn.ensemble import AdaBoostClassifier
    from xgboost import XGBClassifier

    X, y = _tiny_dataset()
    return {
        "xgboost": XGBClassifier(n_estimators=10, max_depth=2, random_state=0).fit(X, y),
        "adaboost": AdaBoostClassifier(n_estimators=10, random_state=0).fit(X, y),
    }


def test_fold_importance_xgboost_uses_tree_explainer(tiny_models):
    sa = _shap_analysis()
    X, _ = _tiny_dataset()

    importance, metadata = sa.compute_fold_shap_importance(
        model=tiny_models["xgboost"],
        model_key="xgboost",
        X_train=X.iloc[:80],
        X_validation=X.iloc[80:],
        max_explain_samples=30,
        background_samples=20,
        random_state=0,
    )

    assert list(importance.columns) == ["feature", "mean_abs_shap", "rank"]
    assert len(importance) == 4
    assert sorted(importance["rank"]) == [1, 2, 3, 4]  # unique deterministic ranks
    assert importance["mean_abs_shap"].is_monotonic_decreasing
    assert (importance["mean_abs_shap"] >= 0).all()
    assert importance.iloc[0]["feature"] == "f0"  # controlled signal feature wins
    assert metadata["explainer"] == "TreeExplainer"
    assert metadata["n_explained"] == 30
    assert metadata["n_background"] == 0


def test_fold_importance_adaboost_uses_permutation_explainer(tiny_models):
    sa = _shap_analysis()
    X, _ = _tiny_dataset()

    importance, metadata = sa.compute_fold_shap_importance(
        model=tiny_models["adaboost"],
        model_key="adaboost",
        X_train=X.iloc[:80],
        X_validation=X.iloc[80:100],
        max_explain_samples=10,
        background_samples=30,
        random_state=0,
    )

    assert len(importance) == 4
    assert metadata["explainer"] == "PermutationExplainer"
    assert metadata["n_background"] == 30
    assert importance.iloc[0]["feature"] == "f0"


def test_fold_importance_rejects_unknown_model_key(tiny_models):
    sa = _shap_analysis()
    X, _ = _tiny_dataset()

    with pytest.raises(ValueError, match="Unsupported model key"):
        sa.compute_fold_shap_importance(
            model=tiny_models["xgboost"],
            model_key="knn",
            X_train=X, X_validation=X,
            max_explain_samples=10, background_samples=10, random_state=0,
        )


# ---------------------------------------------------------------------------
# compute_pairwise_rank_stability (hand-computed)
# ---------------------------------------------------------------------------
def _importance_frame(ranks_by_fold: dict[int, dict[str, int]]) -> pd.DataFrame:
    rows = []
    for fold, ranks in ranks_by_fold.items():
        for feature, rank in ranks.items():
            rows.append({
                "dataset_key": "ds", "dataset_name": "Dataset",
                "model_key": "m", "model_name": "Model",
                "fold": fold, "feature": feature, "rank": rank,
            })
    return pd.DataFrame(rows)


def test_pairwise_stability_identical_folds():
    sa = _shap_analysis()
    frame = _importance_frame({1: {"a": 1, "b": 2, "c": 3}, 2: {"a": 1, "b": 2, "c": 3}})

    pairwise = sa.compute_pairwise_rank_stability(frame, top_k=2)

    assert len(pairwise) == 1
    assert pairwise.iloc[0]["spearman_rho"] == pytest.approx(1.0)
    assert pairwise.iloc[0]["top_k_jaccard"] == pytest.approx(1.0)


def test_pairwise_stability_reversed_folds():
    sa = _shap_analysis()
    frame = _importance_frame({1: {"a": 1, "b": 2, "c": 3}, 2: {"c": 1, "b": 2, "a": 3}})

    pairwise = sa.compute_pairwise_rank_stability(frame, top_k=2)

    assert pairwise.iloc[0]["spearman_rho"] == pytest.approx(-1.0)
    # top2 {a,b} vs {c,b}: intersection 1, union 3
    assert pairwise.iloc[0]["top_k_jaccard"] == pytest.approx(1 / 3)


def test_pairwise_stability_partial_agreement_hand_computed():
    sa = _shap_analysis()
    frame = _importance_frame({1: {"a": 1, "b": 2, "c": 3}, 2: {"a": 1, "b": 3, "c": 2}})

    pairwise = sa.compute_pairwise_rank_stability(frame, top_k=2)

    # ranks [1,2,3] vs [1,3,2]: rho = 1 - 6*2/(3*8) = 0.5; top2 {a,b} vs {a,c}
    assert pairwise.iloc[0]["spearman_rho"] == pytest.approx(0.5)
    assert pairwise.iloc[0]["top_k_jaccard"] == pytest.approx(1 / 3)


def test_pairwise_stability_covers_all_fold_pairs():
    sa = _shap_analysis()
    frame = _importance_frame({f: {"a": (f % 3) + 1, "b": 2, "c": 3} for f in range(1, 6)})

    pairwise = sa.compute_pairwise_rank_stability(frame, top_k=2)

    assert len(pairwise) == 10  # C(5, 2)
    assert set(pairwise["spearman_rho"].between(-1, 1))
    assert set(pairwise["top_k_jaccard"].between(0, 1))


def test_pairwise_stability_rejects_invalid_input():
    sa = _shap_analysis()
    frame = _importance_frame({1: {"a": 1, "b": 2}, 2: {"a": 1, "b": 2}})

    with pytest.raises(ValueError, match="required SHAP-importance columns"):
        sa.compute_pairwise_rank_stability(frame.drop(columns=["rank"]), top_k=2)
    with pytest.raises(ValueError, match="top_k must be positive"):
        sa.compute_pairwise_rank_stability(frame, top_k=0)
    with pytest.raises(ValueError, match="Feature sets differ"):
        mismatched = _importance_frame({1: {"a": 1, "b": 2}, 2: {"a": 1, "c": 2}})
        sa.compute_pairwise_rank_stability(mismatched, top_k=2)


# ---------------------------------------------------------------------------
# summarize_rank_stability / summarize_consensus_ranking
# ---------------------------------------------------------------------------
def test_summarize_rank_stability_aggregates_pairs():
    sa = _shap_analysis()
    frame = _importance_frame({f: {"a": (f % 3) + 1, "b": 2, "c": 3} for f in range(1, 6)})
    pairwise = sa.compute_pairwise_rank_stability(frame, top_k=2)

    summary = sa.summarize_rank_stability(pairwise)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["pair_count"] == 10
    for column in ("spearman_mean", "spearman_std", "spearman_min", "spearman_max",
                   "top_k_jaccard_mean", "top_k_jaccard_std"):
        assert column in summary.columns
    assert row["spearman_min"] <= row["spearman_mean"] <= row["spearman_max"]


def test_summarize_consensus_ranking_hand_computed():
    sa = _shap_analysis()
    # feature a: ranks 1,1,2 across folds; b: 2,2,1; c: 3,3,3
    frame = _importance_frame({
        1: {"a": 1, "b": 2, "c": 3},
        2: {"a": 1, "b": 2, "c": 3},
        3: {"b": 1, "a": 2, "c": 3},
    })
    frame["mean_abs_shap"] = 1.0

    consensus = sa.summarize_consensus_ranking(frame, top_k=2)

    by_feature = consensus.set_index("feature")
    assert by_feature.loc["a", "mean_rank"] == pytest.approx((1 + 1 + 2) / 3)
    # a sits inside the top-2 of every fold (ranks 1, 1, 2)
    assert by_feature.loc["a", "top_k_frequency"] == pytest.approx(1.0)
    assert by_feature.loc["c", "top_k_frequency"] == 0.0
    assert by_feature.loc["a", "consensus_rank"] == 1  # lowest mean rank wins
    assert by_feature.loc["b", "consensus_rank"] == 2
    assert (consensus["folds_observed"] == 3).all()


# ---------------------------------------------------------------------------
# run_shap_experiments end-to-end on a temporary project
# ---------------------------------------------------------------------------
def _write_e2e_project(root: Path) -> Path:
    import yaml

    rng = np.random.default_rng(1)
    n = 160
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    y = (2.0 * X.f0 + rng.normal(0, 0.5, n) > 0).astype(int)
    cut = 120
    for name, frame_X, frame_y in (("train", X[:cut], y[:cut]), ("test", X[cut:], y[cut:])):
        path = root / "data" / "processed" / "qa1"
        path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(frame_X, columns=X.columns).assign(target=frame_y).to_csv(
            path / f"{name}.csv", index=False
        )

    config = {
        "experiment": {
            "random_state": 0,
            "decision_threshold": 0.5,
            "primary_metric": "pr_auc",
            "balance_training": True,
            "output_dir": "results/model_evaluation",
            "cv": {"n_splits": 3, "shuffle": True},
        },
        "datasets": {
            "qa1": {
                "name": "QA Dataset",
                "train_path": "data/processed/qa1/train.csv",
                "test_path": "data/processed/qa1/test.csv",
                "target_column": "target",
            }
        },
        "models": {
            "adaboost": {"display_name": "AdaBoost", "params": {"n_estimators": 10}},
            "xgboost": {"display_name": "XGBoost", "params": {"n_estimators": 10, "max_depth": 2}},
            "lightgbm": {"display_name": "LightGBM", "params": {"n_estimators": 10, "max_depth": 2}},
        },
        "shap": {
            "output_dir": "results/shap_qa",
            "max_explain_samples": 15,
            "background_samples": 25,
            "top_k": 2,
        },
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_run_shap_experiments_end_to_end_output_contract(tmp_path):
    run_shap = _run_shap()
    config_path = _write_e2e_project(tmp_path)

    paths = run_shap.run_shap_experiments(config_path)
    by_name = {Path(path).name: Path(path) for path in paths.values()}

    assert paths and all(path.exists() for path in by_name.values())
    assert {
        "shap_feature_importance.csv", "shap_fold_stability.csv",
        "shap_consensus_ranking.csv", "shap_stability_summary.csv",
        "shap_metadata.json",
    }.issubset(by_name)

    feature_importance = pd.read_csv(by_name["shap_feature_importance.csv"])
    assert len(feature_importance) == 3 * 3 * 4  # models x folds x features
    for column in ("dataset_key", "model_key", "fold", "feature", "rank", "mean_abs_shap"):
        assert column in feature_importance.columns
    assert (feature_importance["mean_abs_shap"] >= 0).all()

    pairwise = pd.read_csv(by_name["shap_fold_stability.csv"])
    assert len(pairwise) == 3 * 3  # models x C(3, 2) fold pairs
    assert pairwise["spearman_rho"].between(-1, 1).all()
    assert pairwise["top_k_jaccard"].between(0, 1).all()

    consensus = pd.read_csv(by_name["shap_consensus_ranking.csv"])
    assert len(consensus) == 3 * 4  # models x features
    for _, group in consensus.groupby("model_key"):
        assert sorted(group["consensus_rank"]) == [1, 2, 3, 4]

    metadata = json.loads(by_name["shap_metadata.json"].read_text(encoding="utf-8"))
    assert isinstance(metadata, dict)


def test_run_shap_requires_shap_config_section(tmp_path):
    run_shap = _run_shap()
    config_path = _write_e2e_project(tmp_path)
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.pop("shap")
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="'shap' configuration"):
        run_shap.run_shap_experiments(config_path)


# ---------------------------------------------------------------------------
# Shipped result artifacts inside the PR tree
# ---------------------------------------------------------------------------
def test_shipped_shap_artifacts_are_well_formed():
    results = REPO_ROOT / "results" / "shap_explainability"
    if not (results / "shap_feature_importance.csv").exists():
        pytest.skip("PR #8 shipped artifacts not present on this branch")

    importance = pd.read_csv(results / "shap_feature_importance.csv")
    grouped = importance.groupby(["dataset_key", "model_key", "fold"])["rank"]
    assert (grouped.apply(lambda ranks: sorted(ranks) == list(range(1, len(ranks) + 1)))).all()
    assert (importance["mean_abs_shap"] >= 0).all()

    pairwise = pd.read_csv(results / "shap_fold_stability.csv")
    assert pairwise["spearman_rho"].between(-1, 1).all()
    assert pairwise["top_k_jaccard"].between(0, 1).all()
    # 3 datasets x 3 models x C(5, 2) pairs
    assert len(pairwise) == 9 * 10

    summary = pd.read_csv(results / "shap_stability_summary.csv")
    assert len(summary) == 9  # one row per dataset x model group

    metadata = json.loads((results / "shap_metadata.json").read_text(encoding="utf-8"))
    assert isinstance(metadata, dict)
