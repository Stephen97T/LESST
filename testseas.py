from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import (
    LinearRegression,
    HuberRegressor,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from main import run_LESST, performance_LESST, benchmark
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

dataset = "Yearly"  # , "Monthly", "Weekly", "Daily", "Hourly"]
frequency = 1  # , 12, 52, 7, 24]
n_cluster = 10
deseason = False

localmodel = models["huber"]
globalmodel = models["huber"]
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
)

lesst_owa, lesst_smape, lesst_mase, lesst_rmse, = performance_LESST(
    predictions,
    dataset,
    res_train,
    res_test,
    horizon,
    frequency,
)

local = less.LocalM
clusters = less.df.cluster
train = less.val
localpred = []
inputs = np.array(last_values(train, timesteps=horizon))
for i, y in enumerate(inputs):
    cluster = clusters[i]
    predic = local.predict(y.reshape(1, -1), cluster=cluster)
    localpred.append(predic)
localpred = np.array(localpred).reshape(predictions.shape)

local_owa, local_smape, local_mase, local_rmse, = performance_LESST(
    localpred,
    dataset,
    res_train,
    res_test,
    horizon,
    frequency,
)
"""
(benchmark_owa, benchmark_smape, benchmark_mase, benchmark_rmse,) = benchmark(
    predictions,
    dataset,
    res_train,
    res_test,
    horizon,
    frequency,
)
"""
