from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import SPLIT_SEED, TEST_SIZE


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the dataset after its canonical filename and format are chosen."""
    raise NotImplementedError("Dataset loading will be added after the dataset is selected.")


def create_development_test_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into development and held-out test sets."""
    development, test = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=SPLIT_SEED,
    )
    return development, test
