import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Eventual result-table columns:
# target, model, n_features, fold, model_seed, mae, rmse, r2,
# train_time, prediction_time
def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return MAE, RMSE, and R-squared for regression predictions."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }
