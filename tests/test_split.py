"""Unit tests for src/data/split.py: train/test split + Stratified K-Fold.

Covers: stratification, train/val/test separation, index disjointness,
per-sample single validation coverage, class proportions across folds,
reproducibility under a fixed seed, and invalid inputs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from config import RANDOM_STATE, TEST_SIZE
from data.split import split_train_test, stratified_kfold_splits


def make_stratified_df(n_per_class: int = 50) -> pd.DataFrame:
	"""Two-class frame with known proportions and a distinguishing feature."""
	return pd.DataFrame(
		{
			"target": [0] * n_per_class + [1] * n_per_class,
			"num": list(range(n_per_class)) + list(range(100, 100 + n_per_class)),
		}
	)


# ---------------------------------------------------------------------------
# split_train_test
# ---------------------------------------------------------------------------

def test_split_sizes_match_test_size():
	df = make_stratified_df()
	split = split_train_test(df, target_column="target", test_size=TEST_SIZE)
	assert len(split.test_df) == int(round(len(df) * TEST_SIZE))
	assert len(split.train_df) == len(df) - len(split.test_df)


def test_split_is_stratified():
	df = make_stratified_df(n_per_class=50)
	split = split_train_test(df, target_column="target")
	overall_ratio = df["target"].mean()
	assert abs(split.train_df["target"].mean() - overall_ratio) < 0.01
	assert abs(split.test_df["target"].mean() - overall_ratio) < 0.05


def test_train_and_test_disjoint():
	df = make_stratified_df()
	split = split_train_test(df, target_column="target")
	train_keys = set(map(tuple, split.train_df[["num"]].to_numpy()))
	test_keys = set(map(tuple, split.test_df[["num"]].to_numpy()))
	assert not (train_keys & test_keys), "sample appears in both train and test"


def test_split_preserves_all_rows():
	df = make_stratified_df()
	split = split_train_test(df, target_column="target")
	assert len(split.train_df) + len(split.test_df) == len(df)


def test_split_reproducible_with_same_seed():
	df = make_stratified_df()
	a = split_train_test(df, target_column="target", random_state=RANDOM_STATE)
	b = split_train_test(df, target_column="target", random_state=RANDOM_STATE)
	pd.testing.assert_frame_equal(a.train_df, b.train_df)
	pd.testing.assert_frame_equal(a.test_df, b.test_df)


def test_split_differs_under_different_seed():
	df = make_stratified_df(n_per_class=200)
	a = split_train_test(df, target_column="target", random_state=1)
	b = split_train_test(df, target_column="target", random_state=2)
	assert not a.train_df["num"].equals(b.train_df["num"])


def test_split_does_not_mutate_input():
	df = make_stratified_df()
	before = df.copy(deep=True)
	split_train_test(df, target_column="target")
	pd.testing.assert_frame_equal(df, before)


# ---------------------------------------------------------------------------
# stratified_kfold_splits
# ---------------------------------------------------------------------------

def test_kfold_generates_configured_fold_count():
	df = make_stratified_df()
	folds = stratified_kfold_splits(df, target_column="target", n_splits=5)
	assert len(folds) == 5


def test_each_sample_validated_exactly_once():
	df = make_stratified_df()
	folds = stratified_kfold_splits(df, target_column="target", n_splits=5)
	val_union = []
	for _, val_idx in folds:
		val_union.extend(val_idx.tolist())
	assert sorted(val_union) == list(range(len(df))), "validation sets must partition the data"


def test_train_val_indices_do_not_overlap_within_fold():
	df = make_stratified_df()
	for train_idx, val_idx in stratified_kfold_splits(df, "target", n_splits=5):
		assert not (set(train_idx) & set(val_idx))


def test_kfold_preserves_class_proportions_across_folds():
	df = pd.DataFrame({"target": [0] * 400 + [1] * 100, "num": range(500)})  # 80/20
	overall_positive = df["target"].mean()
	for _, val_idx in stratified_kfold_splits(df, "target", n_splits=5):
		val_positive = df.iloc[val_idx]["target"].mean()
		assert abs(val_positive - overall_positive) < 0.02


def test_kfold_reproducible_with_same_seed():
	df = make_stratified_df()
	a = stratified_kfold_splits(df, "target", n_splits=5, random_state=RANDOM_STATE)
	b = stratified_kfold_splits(df, "target", n_splits=5, random_state=RANDOM_STATE)
	assert [(list(tr), list(va)) for tr, va in a] == [(list(tr), list(va)) for tr, va in b]


def test_kfold_differs_under_different_seed():
	df = make_stratified_df(n_per_class=300)
	a = stratified_kfold_splits(df, "target", n_splits=5, random_state=1)
	b = stratified_kfold_splits(df, "target", n_splits=5, random_state=2)
	assert list(a[0][1]) != list(b[0][1])


def test_kfold_n_splits_exceeding_minority_class_raises():
	df = pd.DataFrame({"target": [0, 0, 0, 1], "num": [1, 2, 3, 4]})  # only 1 positive
	with pytest.raises(ValueError):
		stratified_kfold_splits(df, "target", n_splits=5)


def test_kfold_rejects_empty_frame():
	with pytest.raises(ValueError):
		stratified_kfold_splits(pd.DataFrame({"target": [], "num": []}), "target", n_splits=5)
