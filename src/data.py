from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURE_COLUMNS, SPLIT_SEED, TARGET_COLUMNS, TEST_SIZE


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "ENB2012_data.xlsx"

COLUMN_MAP = {
    "X1": "relative_compactness",
    "X2": "surface_area",
    "X3": "wall_area",
    "X4": "roof_area",
    "X5": "overall_height",
    "X6": "orientation",
    "X7": "glazing_area",
    "X8": "glazing_area_distribution",
    "Y1": "heating_load",
    "Y2": "cooling_load",
}


def load_data() -> pd.DataFrame:
    """Load the UCI Energy Efficiency dataset with readable column names."""
    df = pd.read_excel(DATA_PATH)
    df = df.rename(columns=COLUMN_MAP)

    expected_columns = FEATURE_COLUMNS + TARGET_COLUMNS
    if df.columns.tolist() != expected_columns:
        raise ValueError(
            "Unexpected dataset columns.\n"
            f"Expected: {expected_columns}\n"
            f"Found: {df.columns.tolist()}"
        )

    return df


def create_dev_test_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create one reproducible development/test split for features and targets."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMNS]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SPLIT_SEED,
    )
