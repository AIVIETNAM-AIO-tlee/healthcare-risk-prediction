from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import N_SPLITS, RANDOM_STATE, TEST_SIZE


@dataclass(frozen=True)
class SplitData:
	train_df: pd.DataFrame
	test_df: pd.DataFrame


def split_train_test(
	df: pd.DataFrame,
	target_column: str,
	test_size: float = TEST_SIZE,
	random_state: int = RANDOM_STATE,
) -> SplitData:
	"""Stratified 80/20 train/test split.

	Splitting happens on the raw (but target-cleaned) dataframe, *before* any
	imputer/encoder/scaler is fitted, so that fitting those transforms only on
	``train_df`` in ``preprocessing.py`` does not leak information from the
	test split.
	"""
	train_df, test_df = train_test_split(
		df,
		test_size=test_size,
		random_state=random_state,
		stratify=df[target_column],
	)
	return SplitData(
		train_df=train_df.reset_index(drop=True),
		test_df=test_df.reset_index(drop=True),
	)


def stratified_kfold_splits(
	train_df: pd.DataFrame,
	target_column: str,
	n_splits: int = N_SPLITS,
	random_state: int = RANDOM_STATE,
) -> list[tuple[pd.Index, pd.Index]]:
	"""Generate shuffled stratified K-Fold (train_idx, val_idx) index pairs.

	Intended for the training split only: the held-out test split must stay
	completely outside this loop. Shuffling + fixed ``random_state`` makes the
	fold assignment reproducible; stratification keeps class proportions
	close across folds. StratifiedKFold itself validates that ``n_splits``
	does not exceed the minority-class size.
	"""
	skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
	return list(skfold.split(train_df, train_df[target_column]))
