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