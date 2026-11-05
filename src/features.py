from functools import partial

from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_regression,
)

from .config import MI_SEED


def make_selector(k):
    """Create an unfitted mutual-information feature selector."""
    mi_score = partial(
        mutual_info_regression,
        random_state=MI_SEED,
    )

    return SelectKBest(
        score_func=mi_score,
        k=k,
    )