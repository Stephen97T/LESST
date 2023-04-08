from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import (
    LinearRegression,
    HuberRegressor,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from main import results_LESST
import numpy as np

np.seed = 1

# models to consider
models = {
    "xgb": XGBRegressor(tree_method="gpu_hist"),
    "lgbm": LGBMRegressor(),
    "rf": RandomForestRegressor(
        bootstrap=True, max_depth=80, max_features=6, n_estimators=100
    ),
    "ols": LinearRegression(),
    "huber": HuberRegressor(),
    "gradient": GradientBoostingRegressor(),
}

datasets = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]
frequencies = [1, 4, 12, 52, 7, 24]
n_clusters = clusters = [3, 10, 30, 50, 100]

localmodels = [
    "ols",
    "huber",
    "xgb",
    "lgbm",
    "huber",
    "lgbm",
]
globalmodels = ["ols", "huber", "xgb", "lgbm", "rf", "huber"]
benchds, lessds = results_LESST(
    datasets,
    n_clusters,
    frequencies,
    localmodels,
    globalmodels,
    models,
    deseason=True,
)

bench, less = results_LESST(
    datasets,
    n_clusters,
    frequencies,
    localmodels,
    globalmodels,
    models,
    deseason=False,
)
