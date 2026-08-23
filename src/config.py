from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Dataset identity
# ---------------------------------------------------------------------------
# Source: Kaggle "Personal Key Indicators of Heart Disease" (2022 update)
# https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
KAGGLE_DATASET_SLUG = "kamilpytlak/personal-key-indicators-of-heart-disease"
TARGET_COLUMN = "HadHeartAttack"

# ---------------------------------------------------------------------------
# Column types
# ---------------------------------------------------------------------------
# The raw 2022 CSV has 6 numeric columns; every other feature column (besides
# the target) is treated as categorical. Listing the numeric columns
# explicitly avoids guessing dtypes from a CSV where numeric columns may still
# contain missing values.
NUMERIC_COLUMNS = [
	"PhysicalHealthDays",
	"MentalHealthDays",
	"SleepHours",
	"HeightInMeters",
	"WeightInKilograms",
	"BMI",
]

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "heart_2022_with_nans.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_CSV_PATH = PROCESSED_DIR / "train.csv"
TEST_CSV_PATH = PROCESSED_DIR / "test.csv"
