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
import pickle

np.seed = 1

# models to consider
models = {
    "huber": HuberRegressor(),
}

datasets = [
    "Yearly",
    "Quarterly",
    "Monthly",
    "Weekly",
    "Daily",
    "Hourly",
]
total_length_timeseries = {}
avg_length_timeseries = {}
test_size = {}
for dataset in datasets:
    test = read_m4test_series(dataset)
    train = read_m4_series(dataset)
    res_train = to_array(train)
    res_test = to_array(test)
    test_size.update({dataset: test.shape[1]})
    count = 0
    for i in res_train:
        count += len(i)
    total_length_timeseries.update({dataset: count})
    avg_length_timeseries.update({dataset: count / len(res_test)})

    print("=====================================================")
    print(f"{dataset} is DONE")
    print("=====================================================")
