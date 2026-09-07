from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from pathlib import Path

from src.experiment_config import load_experiment_config


@dataclass(frozen=True)
class SplitData:
	train_df: pd.DataFrame
	test_df: pd.DataFrame


def split_train_test(
	df: pd.DataFrame,
	target_column: str,
	test_size: float | None = None,
	random_state: int | None = None,
) -> SplitData:
	"""Stratified 80/20 train/test split.

	Splitting happens on the raw (but target-cleaned) dataframe, *before* any
	imputer/encoder/scaler is fitted, so that fitting those transforms only on
	``train_df`` in ``preprocessing.py`` does not leak information from the
	test split.
	"""
	if test_size is None or random_state is None:
		config, _ = load_experiment_config(Path(__file__).resolve().parents[2] / "config.yaml")
		if test_size is None:
			test_size = float(config["preprocessing"]["test_size"])
		if random_state is None:
			random_state = int(config["experiment"]["random_state"])

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
	n_splits: int | None = None,
	random_state: int | None = None,
) -> list[tuple[pd.Index, pd.Index]]:
	"""Generate shuffled stratified K-Fold (train_idx, val_idx) index pairs.

	Intended for the training split only: the held-out test split must stay
	completely outside this loop. Shuffling + fixed ``random_state`` makes the
	fold assignment reproducible; stratification keeps class proportions
	close across folds. StratifiedKFold itself validates that ``n_splits``
	does not exceed the minority-class size.
	"""
	if n_splits is None or random_state is None:
		config, _ = load_experiment_config(Path(__file__).resolve().parents[2] / "config.yaml")
		if n_splits is None:
			n_splits = int(config["experiment"]["cv"]["n_splits"])
		if random_state is None:
			random_state = int(config["experiment"]["random_state"])
	shuffle = True
	if n_splits is not None:
		config, _ = load_experiment_config(Path(__file__).resolve().parents[2] / "config.yaml")
		shuffle = bool(config["experiment"]["cv"].get("shuffle", True))
	skfold = StratifiedKFold(
		n_splits=n_splits,
		shuffle=shuffle,
		random_state=random_state if shuffle else None,
	)
	return list(skfold.split(train_df, train_df[target_column]))
