from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import (
    LinearRegression,
    HuberRegressor,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from models import WeightedSum
from LESST import LESST
from benchmark import BenchmarkModel, PerformanceMeasures
from tsforecast import ThetaF
import numpy as np
from data_prep import to_array
from preprocessing import read_m4test_series, read_m4_series
from time import time

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


datasets = ["quarterly"]
n_clusters = [300]
frequencies = [4]
localmodels = ["ols", "ols", "ols", "ols", "ols", "ols"]
globalmodels = ["ols", "huber", "xgb", "lgbm", "rf", "gradient"]


def run_LESST(
    dataset,
    train,
    n_clusters,
    horizon,
    frequency,
    localmodel,
    globalmodel,
    deseason,
):
    less = LESST(dataset, train, n_clusters, frequency, deseason)
    less.fit(
        prediction_steps=horizon,
        localmodel=localmodel,
        globalmodel=globalmodel,
    )

    predictions = less.predict()
    return predictions, less


def performance_LESST(predictions, dataset, train, test, horizon, frequency):
    lesst_owa = []

    for i in range(0, len(train)):
        measure = PerformanceMeasures(frequency)
        model_owa = measure.OWA(test[i], predictions[i], train[i])
        lesst_owa.append(model_owa)

    lesst_owa = np.array(lesst_owa)
    lesst_owa[lesst_owa >= 1e308] = np.nan
    lesst_owa = np.nan_to_num(lesst_owa).mean()
    return lesst_owa


def benchmark(predictions, dataset, train, test, horizon, frequency):
    total_train = train

    benchmark_owa = []

    for i in range(0, len(train)):
        benchmark = BenchmarkModel(ThetaF(frequency))
        bench_owa = benchmark.performance(
            test[i], train[i], horizon, frequency
        )
        benchmark_owa.append(bench_owa)

    bench = np.array(benchmark_owa)
    bench[bench >= 1e308] = np.nan
    bench = np.nan_to_num(bench).mean()
    return bench


def results_LESST(
    datasets,
    n_clusters,
    frequencies,
    localmodels,
    globalmodels,
    models,
    deseason,
):
    total_lesst_owas = {}
    total_benchmark_owas = {}
    for instance, dataset in enumerate(datasets):
        test = read_m4test_series(dataset)
        train = read_m4_series(dataset)
        res_train = to_array(train)
        res_test = to_array(test)
        lesst_owas = {}
        horizon = test.shape[1]
        frequency = frequencies[instance]
        for n_cluster in n_clusters:
            for localmodelname, globalmodelname in zip(
                localmodels, globalmodels
            ):
                localmodel = models[localmodelname]
                globalmodel = models[globalmodelname]
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
                t = time()
                lesst_owa = performance_LESST(
                    predictions,
                    dataset,
                    res_train,
                    res_test,
                    horizon,
                    frequency,
                )
                print(f"LESST performance calculation time {time()-t} sec")
                lesst_owas.update(
                    {
                        f"ncl:{n_cluster}/lm:{localmodelname}/gm:{globalmodenamel}": lesst_owa,
                    }
                )
        t = time()
        # benchmark_owa = benchmark(
        #    predictions, dataset, res_train, res_test, horizon, frequency
        # )
        benchmark_owa = 1
        print(f"Benchmark performance calculation time {time()-t} sec")
        total_lesst_owas.update({f"ds:{dataset}": lesst_owas})
        total_benchmark_owas.update({f"ds:{dataset}": benchmark_owa})

    return total_benchmark_owas, total_lesst_owas


bench, less = results_LESST(
    datasets,
    n_clusters,
    frequencies,
    localmodels,
    globalmodels,
    models,
    deseason=False,
)
