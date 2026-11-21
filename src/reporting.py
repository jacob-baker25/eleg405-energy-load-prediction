"""Shared helpers for turning fold-level result tables into report tables.

Every results CSV in results/tables/ has one row per target, model, feature
count, and fold (plus a seed column for stochastic models). These helpers give
every model the same summary format, so linear, ridge, and random-forest
results can be placed side by side without reformatting.
"""

import pandas as pd

TIME_COLUMNS = ["selection_time", "tuning_time", "train_time"]

TARGET_LABELS = {
    "heating_load": "Heating",
    "cooling_load": "Cooling",
}


def add_total_fit_time(results: pd.DataFrame) -> pd.DataFrame:
    """Add total_fit_time = selection + tuning + training time (seconds).

    This is the full cost of producing one fitted model on a training fold.
    Missing timing columns (for example, an untuned model without feature
    selection) count as 0.
    """
    results = results.copy()
    for column in TIME_COLUMNS:
        if column not in results:
            results[column] = 0.0
        results[column] = results[column].fillna(0.0)

    results["total_fit_time"] = results[TIME_COLUMNS].sum(axis=1)
    return results


def summarize(
    results: pd.DataFrame,
    metrics=("rmse", "mae", "r2"),
    by=("target", "model", "selector", "n_features"),
) -> pd.DataFrame:
    """Mean and standard deviation of each metric for every configuration.

    The standard deviation is taken over all rows in a group, so it covers fold
    variability (and seed variability when several seeds are present).
    Grouping columns that a table does not have are skipped.
    """
    by = [column for column in by if column in results.columns]

    summary = results.groupby(by, sort=False).agg(
        n_runs=(metrics[0], "size"),
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_std": (m, "std") for m in metrics},
    )

    ordered = ["n_runs"]
    for m in metrics:
        ordered += [f"{m}_mean", f"{m}_std"]

    return summary[ordered].reset_index()


def format_mean_std(summary: pd.DataFrame, metric: str, decimals: int = 2) -> pd.Series:
    """Format one metric from summarize() as 'mean ± std' text."""
    mean = summary[f"{metric}_mean"]
    std = summary[f"{metric}_std"]
    return mean.map(f"{{:.{decimals}f}}".format) + " ± " + std.map(f"{{:.{decimals}f}}".format)


def to_latex_table(table: pd.DataFrame, path, caption: str, label: str) -> str:
    """Write a simple IEEE-style LaTeX table (booktabs) and return its text.

    Cell values are written as they appear in the DataFrame, so format numbers
    first (for example, with format_mean_std). '±' is converted to $\\pm$.
    """
    def clean(value):
        text = str(value).replace("_", r"\_").replace("%", r"\%")
        return text.replace("±", r"$\pm$")

    column_format = "l" * len(table.columns)
    header = " & ".join(clean(c) for c in table.columns) + r" \\"
    body = [" & ".join(clean(v) for v in row) + r" \\" for row in table.itertuples(index=False)]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
        header,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    text = "\n".join(lines) + "\n"

    with open(path, "w", encoding="utf-8") as file:
        file.write(text)

    return text
