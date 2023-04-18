from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import (
    LinearRegression,
    HuberRegressor,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from main import run_LESST, performance_LESST, benchmark, local_results
from data_prep import to_array
from preprocessing import read_m4test_series, read_m4_series
from data_prep import last_values
import numpy as np
from models import WeightedSum

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
]  # , "Monthly", "Weekly", "Daily", "Hourly"]
frequencys = [1, 4, 12, 52, 7, 24]  # , 12, 52, 7, 24]
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
    WeightedSum(),
    WeightedSum(),
    WeightedSum(),
    WeightedSum(),
    WeightedSum(),
    WeightedSum(),
]
lesst_owas = {}
lesst_smapes = {}
lesst_rmses = {}
lesst_mases = {}
for dataset, frequency, n_cluster, deseason, localmodel, globalmodel in zip(
    datasets, frequencys, n_clusters, deseasons, localmodels, globalmodels
):
    test = read_m4test_series(dataset)
    train = read_m4_series(dataset)
    res_train = to_array(train)
    res_test = to_array(test)
    horizon = test.shape[1]
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
        evenweighted=False
    )
    lesst_owa, lesst_smape, lesst_mase, lesst_rmse = performance_LESST(
        predictions,
        dataset,
        res_train,
        res_test,
        horizon,
        frequency,
    )
    lesst_owas.update({f"{dataset}": lesst_owa})
    lesst_smapes.update({f"{dataset}": lesst_smape})
    lesst_mases.update({f"{dataset}": lesst_mase})
    lesst_rmses.update({f"{dataset}": lesst_rmse})

"""