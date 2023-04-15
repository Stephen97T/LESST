from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import (
    LinearRegression,
    HuberRegressor,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from main import run_LESST, performance_LESST
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

dataset = "Yearly"  # , "Monthly", "Weekly", "Daily", "Hourly"]
frequency = 1  # , 12, 52, 7, 24]
n_cluster = 2
deseason = False

localmodel = "huber"
globalmodel = "huber"

predictions, less = run_LESST(
    dataset,
    train,
    n_cluster,
    horizon,
    frequency,
    localmodel,
    globalmodel,
    deseason,
)

lesst_owa, lesst_smape, lesst_mase, lesst_rmse, = performance_LESST(
    predictions,
    dataset,
    res_train,
    res_test,
    horizon,
    frequency,
)
