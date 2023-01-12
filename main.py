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
from data_prep import get_train
from preprocessing import read_m4test_series, read_m4_series

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
n_clusters = [10, 100]
horizons = [4]
frequencies = [4]
localmodels = ["ols"]
globalmodels = ["huber"]


def run_LESST(
    dataset, train, n_clusters, horizon, frequency, localmodel, globalmodel
):
    less = LESST(dataset, train, n_clusters, frequency)
    less.fit(
        prediction_steps=horizon,
        localmodel=localmodel,
        globalmodel=globalmodel,
    )

    predictions = less.predict()
    return predictions, less


def performance_LESST(
    predictions, less, dataset, test, train, horizon, frequency
):
    lesst_owa = []

    for i in range(0, len(train)):
        measure = PerformanceMeasures(frequency)
        model_owa = measure.OWA(test[i], predictions[i], train[i])
        lesst_owa.append(model_owa)

    lesst_owa = np.array(lesst_owa)
    lesst_owa[lesst_owa >= 1e308] = 0
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
    bench[bench >= 1e308] = 0
    bench = np.nan_to_num(bench).mean()
    return bench


def results_LESST(
    datasets,
    n_clusters,
    horizons,
    frequencies,
    localmodels,
    globalmodels,
    models,
):
    total_lesst_owas = {}
    total_benchmark_owas = {}
    for instance, dataset in enumerate(datasets):
        test = read_m4test_series(dataset)
        train = read_m4_series(dataset)
        lesst_owas = {}
        horizon = horizons[instance]
        frequency = frequencies[instance]
        for n_cluster in n_clusters:
            for localmodelname, globalmodelname in zip(
                localmodels, globalmodels
            ):
                localmodel = models[localmodelname]
                globalmodel = models[globalmodelname]
                predictions, less = run_LESST(
                    dataset,
                    n_cluster,
                    horizon,
                    frequency,
                    localmodel,
                    globalmodel,
                )
                lesst_owa = performance_LESST(
                    predictions, less, dataset, test, train, horizon, frequency
                )
                lesst_owas.update(
                    {
                        f"ncl:{n_cluster}/lm:{localmodel}/gm:{globalmodel}": lesst_owa,
                    }
                )

        benchmark_owa = benchmark(
            predictions, dataset, train, test, horizon, frequency
        )
        total_lesst_owas.update({f"ds:{dataset}": lesst_owas})
        total_benchmark_owas.update({f"ds:{dataset}": benchmark_owa})

    return total_benchmark_owas, total_lesst_owas


results_LESST(
    datasets,
    n_clusters,
    horizons,
    frequencies,
    localmodels,
    globalmodels,
    models,
)
