from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2

# ---------------------------------------------------------------------------
# Cross-validation (on the training split only; test split stays outside CV)
# ---------------------------------------------------------------------------
N_SPLITS = 5

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass(frozen=True)
class DatasetConfig:
	"""Everything dataset-specific: where its raw/processed files live, its
	target/feature columns, and any data-quality quirks that need special
	handling during preprocessing.

	Each dataset lives under its own subfolder (``data/raw/<key>/`` and
	``data/processed/<key>/``), matching the on-disk layout under the
	project's ``data/`` folder.
	"""

	key: str
	name: str
	source_url: str
	raw_filename: str
	target_column: str
	numeric_columns: list[str]
	# Columns where 0 is really a "not measured" placeholder rather than a
	# genuine reading (e.g. Cholesterol/RestingBP in dataset2) -- these get
	# converted to NaN before imputation instead of being taken at face value.
	zero_as_missing_columns: list[str] = field(default_factory=list)
	# Whether to drop exact-duplicate rows before cleaning the target (e.g.
	# dataset3, whose mostly-binary/discrete features produce genuine
	# duplicate survey responses).
	drop_duplicate_rows: bool = False
	# Continuous numeric columns to IQR-clip (Tukey fences) for outlier
	# handling, fit on the training split only (see
	# data.preprocessing.fit_outlier_bounds). Empty by default: only
	# dataset2's genuinely continuous clinical measurements opt in (see
	# docs/qa-scope-methodology-review-handoff.md, finding F5) -- the
	# mostly-binary/discrete survey data in dataset1/dataset3 is excluded,
	# since IQR clipping there would cut real disease signal (e.g. a
	# genuinely high BMI) rather than noise. Binary 0/1 flags that happen to
	# be numeric-typed (e.g. dataset2's FastingBS) must also stay excluded:
	# their IQR bounds collapse to a single point and would clip away the
	# entire minority-class signal.
	iqr_outlier_columns: list[str] = field(default_factory=list)

	@property
	def raw_path(self) -> Path:
		return RAW_DATA_DIR / self.key / self.raw_filename

	@property
	def processed_dir(self) -> Path:
		return PROCESSED_DIR / self.key

	@property
	def train_csv_path(self) -> Path:
		return self.processed_dir / "train.csv"

	@property
	def test_csv_path(self) -> Path:
		return self.processed_dir / "test.csv"


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
DATASET1 = DatasetConfig(
	key="dataset1",
	name="Personal Key Indicators of Heart Disease (2022)",
	source_url="https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease",
	raw_filename="heart_2022_with_nans.csv",
	target_column="HadHeartAttack",
	numeric_columns=[
		"PhysicalHealthDays",
		"MentalHealthDays",
		"SleepHours",
		"HeightInMeters",
		"WeightInKilograms",
		"BMI",
	],
)

DATASET2 = DatasetConfig(
	key="dataset2",
	name="Heart Failure Prediction",
	source_url="https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction",
	raw_filename="heart.csv",
	target_column="HeartDisease",
	numeric_columns=["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"],
	zero_as_missing_columns=["Cholesterol", "RestingBP"],
	# FastingBS is deliberately excluded: it is a 0/1 flag, not a continuous
	# measurement (see the field docstring above).
	iqr_outlier_columns=["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"],
)

DATASET3 = DatasetConfig(
	key="dataset3",
	name="Heart Disease Health Indicators (BRFSS 2015)",
	source_url="https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset",
	raw_filename="heart_disease_health_indicators_BRFSS2015.csv",
	target_column="HeartDiseaseorAttack",
	numeric_columns=["BMI", "MentHlth", "PhysHlth", "GenHlth", "Age", "Education", "Income"],
	drop_duplicate_rows=True,
)

DATASETS: list[DatasetConfig] = [DATASET1, DATASET2, DATASET3]
DATASETS_BY_KEY: dict[str, DatasetConfig] = {dataset.key: dataset for dataset in DATASETS}
