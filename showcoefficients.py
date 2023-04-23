from sklearn.linear_model import HuberRegressor
from main import run_LESST
from data_prep import to_array
from preprocessing import read_m4test_series, read_m4_series
import numpy as np

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
frequencys = [1, 4, 12, 52, 7, 24]
n_clusters = [10] * 6
deseasons = [False] * 6

localmodels = [models["huber"]] * 6
globalmodels = [models["huber"]] * 6

model_coef = {}

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
    )
    model_coef.update({dataset: list(less.Gmodel.model.coef_)})

    print("=====================================================")
    print(f"{dataset} is DONE")
    print("=====================================================")
