"""This file produces results for the Local Models"""
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import (
    LinearRegression,
    HuberRegressor,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from main import run_LESST, local_results
from data_prep import to_array
from preprocessing import read_m4test_series, read_m4_series
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

datasets = [
    "Yearly",
    "Quarterly",
    "Monthly",
    "Weekly",
    "Daily",
    "Hourly",
]

# Define the parameters used for the best LESST models
frequencys = [1, 4, 12, 52, 7, 24]
n_clusters = [100, 3, 3, 30, 30, 100]
deseasons = [False, True, False, True, False, True]

localmodels = [
    models["huber"],
    models["huber"],
    models["huber"],
    models["huber"],
    models["huber"],
    models["huber"],
]
globalmodels = [
    models["huber"],
    models["huber"],
    models["huber"],
    models["rf"],
    models["huber"],
    models["rf"],
]
local_owas = {}
local_smapes = {}
local_rmses = {}
local_mases = {}
for dataset, frequency, n_cluster, deseason, localmodel, globalmodel in zip(
    datasets, frequencys, n_clusters, deseasons, localmodels, globalmodels
):
    # Initialize data
    test = read_m4test_series(dataset)
    train = read_m4_series(dataset)
    res_train = to_array(train)
    res_test = to_array(test)
    horizon = test.shape[1]

    # Train LESST
    predictions, less = run_LESST(
        dataset,
        train,
        n_cluster,
        horizon,
        frequency,
        localmodel,
        globalmodel,
        deseason,
        rolling=False,
    )

    # Use LESST Local Models to calculate Local performance
    local_owa, local_smape, local_mase, local_rmse = local_results(
        less, dataset, res_train, res_test, predictions, deseason
    )
    local_owas.update({f"{dataset}": local_owa})
    local_smapes.update({f"{dataset}": local_smape})
    local_mases.update({f"{dataset}": local_mase})
    local_rmses.update({f"{dataset}": local_rmse})
