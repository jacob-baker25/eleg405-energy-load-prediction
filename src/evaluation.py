from time import perf_counter

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

from .config import N_FOLDS, CV_SEED, TIMING_REPEATS

# Result-table columns:
# target, model, n_features, fold, model_seed, mae, rmse, r2,
# train_time, prediction_time, selection_time, tuning_time
#
# Timing definitions (seconds):
#   train_time      - fitting the final model on its selected features only
#                     (median of TIMING_REPEATS runs)
#   prediction_time - predicting the validation fold with that model
#                     (median of TIMING_REPEATS runs)
#   selection_time  - mutual-information feature ranking (0 when all 8 are used)
#   tuning_time     - hyperparameter search (0 for untuned models)


def median_time(func, repeats=TIMING_REPEATS) -> float:
    """Run func() several times and return the median elapsed time in seconds."""
    times = []
    for _ in range(repeats):
        start = perf_counter()
        func()
        times.append(perf_counter() - start)
    return float(np.median(times))


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return MAE, RMSE, and R-squared for regression predictions."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }

def get_cv(seed=CV_SEED):
    """Shuffled K-fold splitter. The default seed gives the shared project folds."""
    return KFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=seed,
    )


def get_group_cv(seed=CV_SEED):
    """Shuffled K-fold splitter that keeps each group in a single fold.

    Use with src.data.make_design_groups so the four orientation copies of a
    building design never appear in both training and validation:
        get_group_cv(seed).split(X, groups=groups)
    """
    return GroupKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=seed,
    )
