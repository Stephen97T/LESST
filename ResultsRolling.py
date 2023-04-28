"""This file produces results for when the Global Model uses all data in the train set for
fitting the Global Model"""
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import (
    LinearRegression,
    HuberRegressor,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from main import run_LESST, performance_LESST
from data_prep import to_array
from preprocessing import read_m4test_series, read_m4_series
import numpy as np
import pickle

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
frequencys = [1, 4, 12, 52, 7, 24]
n_clusters = [100, 3, 3, 30, 30, 100]
deseasons = [False, True]

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
    models["huber"],
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
        rolling=True,
    )
    del less
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
    with open(
        f"E:/documents/work/thesis/rolling_{dataset}.pkl",
        "wb",
    ) as handle:
        pickle.dump(
            {
                "owa": lesst_owas,
                "smape": lesst_smapes,
                "mase": lesst_mases,
                "rmse": lesst_rmses,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print("=====================================================")
    print(f"{dataset} is DONE")
    print("=====================================================")
