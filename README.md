# ELEG 405 — Building Energy Load Prediction

## Project

Effects of Model Complexity and Feature Availability on Building Energy-Load Prediction

Jacob Baker  
Nathan Castro

## Research Question

How do regression-model complexity and the amount of available architectural information affect the accuracy and computational efficiency of building heating- and cooling-load prediction?

## Models

- Linear Regression
- Ridge Regression
- Random Forest Regression

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Then activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project requirements:

```bash
python -m pip install -r requirements.txt
```

## Collaboration

Each person should work on their own branch to avoid interfering with the other's work.

**Jacob:** `jacob-work` — data preparation, EDA, shared evaluation tools, linear regression, and ridge regression.

**Nate:** `nate-work` — random forest, tuning, timing, and model-comparison experiments.

After cloning the repository, create your branch:

```bash
git switch -c nate-work
git push -u origin nate-work
```

Before starting a new session, pull the latest changes to `main` and merge them into your working branch:

```bash
git switch main
git pull
git switch nate-work
git merge main
```

If the branch already exists locally, use `git switch nate-work` instead of creating it again.

Keep `main` as the stable, working version. Merge completed work through GitHub pull requests, and avoid editing each other's notebooks without coordinating.

## Shared Experiment Setup

- Dataset: UCI Energy Efficiency (`data/raw/ENB2012_data.xlsx`)
- Split: 80% development / 20% held-out test, seed 42
- Cross-validation: shuffled 5-fold CV on development data, seed 42
- Targets: heating and cooling load, modeled separately
- Feature counts: 2, 4, 6, 8
- Feature selection: mutual information, ranked separately for each target, using `src/features.py`
- Feature selection must be fitted inside training folds, never on the complete development set before CV.
- Sensitivity check: linear regression is also run with forward selection, using `make_forward_selector` in `src/features.py`
- Linear regression: no scaling
- Ridge: `StandardScaler`, with alpha selected by inner 3-fold CV from `[0.01, 0.1, 1, 10, 100]`
- Random forest: no scaling; tuning handled in Nate's notebook
- Metrics: MAE, RMSE, R², training time, prediction time, feature-selection time, tuning time
- Timing: use `median_time` from `src/evaluation.py` (median of 20 runs); column definitions are at the top of that file
- Robustness: repeated CV with shuffle seeds 0–4, and design-grouped CV that keeps the orientation copies of each building design in the same fold (`get_group_cv` and `make_design_groups`)
- Do not use the held-out test set until the final model-selection decisions are complete.

Results are saved to `results/tables/`, with one row per target, model, feature count, and outer fold. Stochastic model runs also identify their random seed.

## Running the Experiments

With the virtual environment activated, run the notebooks in `notebooks/` in order:

1. `01_data_eda.ipynb` — data loading, checks, and exploratory analysis
2. `02_linear_ridge_jacob.ipynb` — linear and ridge feature ablation, forward-selection comparison
3. `03_random_forest_nate.ipynb` — random forest experiments
4. `04_robustness_jacob.ipynb` — repeated and design-grouped CV for linear and ridge (about 12 minutes)
5. `05_linear_diagnostics_jacob.ipynb` — linear failure analysis and ridge alpha sweep
6. `06_results_summary.ipynb` — combined tables and figures from the saved results

Notebook 06 only reads `results/tables/`, so re-run it after any model notebook updates its results. To include a new results file, add it to `RESULT_FILES` in that notebook.

To run a notebook without opening Jupyter:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/05_linear_diagnostics_jacob.ipynb
```

Figures are saved to `results/figures/`. LaTeX versions of the summary tables are saved as `.tex` files in `results/tables/`.


## Checkpoint 2 — Experimental Results and Analysis

Checkpoint 2 expands the initial model comparisons with feature ablation, hyperparameter tuning, computational performance measurements, and error analysis. All experiments use the development set for cross-validation, while the held-out test set remains unused during model selection.

### Random Forest Experiments

The Random Forest experiments are implemented in `notebooks/03_random_forest_nate.ipynb`.

- **Baseline:** 100-tree Random Forest evaluated across five cross-validation folds and five random seeds.
- **Feature Ablation:** Compared 2, 4, 6, and 8 input features using fixed-order and mutual-information feature selection.
- **Hyperparameter Tuning:** Evaluated 12 configurations per target using different numbers of trees, maximum depths, and minimum leaf sizes.
- **Model Comparison:** Compared Random Forest with Linear Regression and Ridge Regression.
- **Computational Performance:** Recorded model training and prediction times.
- **Failure Analysis:** Examined prediction errors across heating- and cooling-load quartiles.

### Principal Results

Random Forest achieved the lowest cross-validation prediction errors among the three models.

| Model | Heating RMSE | Cooling RMSE |
|---|---:|---:|
| Linear Regression | 2.9158 | 3.2131 |
| Ridge Regression | 2.9157 | 3.2134 |
| Random Forest | 0.4903 | 1.6914 |

These values represent the lowest mean validation RMSE for each model across the tested feature counts. Random Forest performed best using six mutual-information-selected features.

Hyperparameter tuning selected 200-tree configurations, with validation RMSE values of 0.5152 for heating and 1.7106 for cooling using eight features and one fixed model seed.

Random Forest required more training time than the simpler regression models but achieved substantially lower prediction error.

### Checkpoint 2 Output Files

**Results Tables:** `results/tables/`

- `random_forest_mi_ablation_cv.csv`
- `random_forest_mi_ablation_summary.csv`
- `random_forest_tuning_cv.csv`
- `random_forest_tuning_summary.csv`
- `random_forest_best_configs.csv`
- `checkpoint2_model_comparison.csv`
- `checkpoint2_timing_comparison.csv`
- `checkpoint2_rf_error_by_range.csv`

**Results Figures:** `results/figures/`

- `checkpoint2_heating_load_comparison.png`
- `checkpoint2_cooling_load_comparison.png`
- `checkpoint2_heating_load_training_time.png`
- `checkpoint2_cooling_load_training_time.png`
- `checkpoint2_heating_load_error_analysis.png`
- `checkpoint2_cooling_load_error_analysis.png`

### Reproducing Checkpoint 2 Results

1. Install dependencies using `pip install -r requirements.txt`.
2. Run `notebooks/03_random_forest_nate.ipynb`.
3. Run `notebooks/06_results_summary.ipynb` to regenerate combined results.
4. Review the generated CSV files and figures in the `results/` directory.

Random Forest timing uses wall-clock measurements, while some Linear Regression and Ridge Regression timings use repeated measurements. Feature-selection and tuning costs should be considered separately.

### Remaining Work

Before the final IEEE paper, the remaining work includes verifying reproducibility from a clean environment, completing additional robustness comparisons if needed, and evaluating the held-out test set after the model configurations have been fixed.
