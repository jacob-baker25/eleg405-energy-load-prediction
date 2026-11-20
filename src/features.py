from functools import partial

from sklearn.feature_selection import (
    SelectKBest,
    SequentialFeatureSelector,
    mutual_info_regression,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from .config import CV_SEED, INNER_CV_FOLDS, MI_SEED


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


def make_forward_selector(k, estimator=None):
    """Create an unfitted forward-selection feature selector.

    Starting from no features, it repeatedly adds the feature that most lowers
    inner-cross-validation RMSE, given the features already chosen. Unlike the
    mutual-information selector, features are judged jointly, so a feature that
    duplicates information already selected adds little and is skipped.
    """
    if estimator is None:
        estimator = LinearRegression()

    return SequentialFeatureSelector(
        estimator,
        n_features_to_select=k,
        direction="forward",
        scoring="neg_root_mean_squared_error",
        cv=KFold(
            n_splits=INNER_CV_FOLDS,
            shuffle=True,
            random_state=CV_SEED,
        ),
    )