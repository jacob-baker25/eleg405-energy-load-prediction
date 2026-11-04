TEST_SIZE = 0.20
SPLIT_SEED = 42
N_FOLDS = 5

CV_SEED = 42

RF_SEEDS = [0, 1, 2, 3, 4]

FEATURE_COLUMNS = [
    "relative_compactness",
    "surface_area",
    "wall_area",
    "roof_area",
    "overall_height",
    "orientation",
    "glazing_area",
    "glazing_area_distribution",
]

TARGET_COLUMNS = [
    "heating_load",
    "cooling_load",
]
