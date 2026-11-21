"""Reusable single-fold evaluations for the linear and ridge experiments.

Each function trains on one training fold, evaluates on the matching validation
fold, and returns one result row (metrics + timings). Experiment loops in the
notebooks only choose the folds and collect the rows, so the same procedure can
be run under different cross-validation schemes without copying code.

These reproduce the feature-ablation loops in notebooks/02_linear_ridge_jacob.ipynb.
"""

from time import perf_counter

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CV_SEED, FEATURE_COLUMNS, INNER_CV_FOLDS, RIDGE_ALPHAS
from .evaluation import median_time, regression_metrics
from .features import make_forward_selector, make_selector


def select_features(X_train, y_train, k, method="mi"):
    """Choose k features using the training fold only.

    method: "mi" (mutual-information ranking) or "forward" (forward selection).
    Returns (selected column names, selection time in seconds). With all
    features requested, nothing is selected and the time is 0.
    """
    if k >= len(FEATURE_COLUMNS):
        return list(X_train.columns), 0.0

    if method == "mi":
        selector = make_selector(k)
    elif method == "forward":
        selector = make_forward_selector(k)
    else:
        raise ValueError(f"Unknown selection method: {method}")

    start = perf_counter()
    selector.fit(X_train, y_train)
    selection_time = perf_counter() - start

    return list(X_train.columns[selector.get_support()]), selection_time


def evaluate_linear(X_train, y_train, X_val, y_val, k, method="mi"):
    """Select k features, fit linear regression, and score the validation fold."""
    selected, selection_time = select_features(X_train, y_train, k, method)

    X_train_k = X_train[selected]
    X_val_k = X_val[selected]

    model = LinearRegression()
    train_time = median_time(lambda: model.fit(X_train_k, y_train))
    prediction_time = median_time(lambda: model.predict(X_val_k))
    predictions = model.predict(X_val_k)

    return {
        "model": "linear",
        "selector": method,
        "n_features": k,
        **regression_metrics(y_val, predictions),
        "train_time": train_time,
        "prediction_time": prediction_time,
        "selection_time": selection_time,
        "tuning_time": 0.0,
        "selected_features": ", ".join(selected),
    }


def make_ridge_pipeline(k):
    """Feature selection (if k < 8), standardization, and Ridge."""
    steps = []

    if k < len(FEATURE_COLUMNS):
        steps.append(("selector", make_selector(k)))

    steps.extend([
        ("scaler", StandardScaler()),
        ("model", Ridge()),
    ])

    return Pipeline(steps)


def evaluate_ridge(X_train, y_train, X_val, y_val, k):
    """Tune alpha by inner CV, fit ridge on MI-selected features, and score.

    Alpha is chosen by GridSearchCV on the training fold, with MI selection and
    scaling re-fit inside every inner fold (fully nested). Predictions come from
    that search; selection and final-model timings are measured separately.
    """
    search = GridSearchCV(
        estimator=make_ridge_pipeline(k),
        param_grid={"model__alpha": RIDGE_ALPHAS},
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=CV_SEED),
        refit=True,
    )

    start = perf_counter()
    search.fit(X_train, y_train)
    tuning_time = perf_counter() - start

    predictions = search.predict(X_val)
    best_alpha = search.best_params_["model__alpha"]

    selected, selection_time = select_features(X_train, y_train, k, "mi")
    X_train_k = X_train[selected]
    X_val_k = X_val[selected]

    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=best_alpha)),
    ])
    train_time = median_time(lambda: final_model.fit(X_train_k, y_train))
    prediction_time = median_time(lambda: final_model.predict(X_val_k))

    return {
        "model": "ridge_tuned",
        "selector": "mi",
        "n_features": k,
        **regression_metrics(y_val, predictions),
        "train_time": train_time,
        "prediction_time": prediction_time,
        "selection_time": selection_time,
        "tuning_time": tuning_time,
        "best_alpha": best_alpha,
        "inner_cv_rmse": -search.best_score_,
        "selected_features": ", ".join(selected),
    }
