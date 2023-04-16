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
import pickle


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
    less = LESST(
        dataset,
        train,
        n_clusters,
        frequency,
        deseason,
        split=True,
        start=True,
        rolling=False,
    )
    less.fit(
        prediction_steps=horizon,
        localmodel=localmodel,
        globalmodel=globalmodel,
    )

    predictions = less.predict()
    return predictions, less


def performance_LESST(predictions, dataset, train, test, horizon, frequency):
    measure = PerformanceMeasures(frequency)
    lesst_owa, lesst_smape, lesst_mase, lesst_rmse = measure.OWA(
        test, predictions, train
    )
    return lesst_owa, lesst_smape, lesst_mase, lesst_rmse


def benchmark(predictions, dataset, train, test, horizon, frequency):
    benchmark = BenchmarkModel(ThetaF(frequency))
    bench_owa, bench_smape, bench_mase, bench_rmse = benchmark.performance(
        test, train, horizon, frequency
    )
    return bench_owa, bench_smape, bench_mase, bench_rmse


def results_LESST(
    datasets,
    n_clusters,
    frequencies,
    localmodels,
    globalmodels,
    models,
    deseason,
    check_benchmark=True,
):
    total_lesst_owas = {}
    total_benchmark_owas = {}
    for instance, dataset in enumerate(datasets):
        test = read_m4test_series(dataset)
        train = read_m4_series(dataset)
        res_train = to_array(train)
        res_test = to_array(test)
        lesst_owas = {}
        lesst_smapes = {}
        lesst_mases = {}
        lesst_rmses = {}
        horizon = test.shape[1]
        frequency = frequencies[instance]
        for n_cluster in n_clusters:
            for localmodelname, globalmodelname in zip(
                localmodels, globalmodels
            ):
                try:
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
                    with open(
                        f"E:/documents/work/thesis/LESSMODEL_ncl_{n_cluster}_lm_{localmodelname}_gm_{globalmodelname}_ds_{deseason}.pkl",
                        "wb",
                    ) as handle:
                        pickle.dump(
                            less,
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                    t = time()
                    (
                        lesst_owa,
                        lesst_smape,
                        lesst_mase,
                        lesst_rmse,
                    ) = performance_LESST(
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
                            f"ncl_{n_cluster}_lm_{localmodelname}_gm_{globalmodelname}_OWA": lesst_owa,
                        }
                    )
                    lesst_smapes.update(
                        {
                            f"ncl_{n_cluster}_lm_{localmodelname}_gm_{globalmodelname}_SMAPE": lesst_smape,
                        }
                    )
                    lesst_mases.update(
                        {
                            f"ncl_{n_cluster}_lm_{localmodelname}_gm_{globalmodelname}_MASE": lesst_mase,
                        }
                    )
                    lesst_rmses.update(
                        {
                            f"ncl_{n_cluster}_lm_{localmodelname}_gm_{globalmodelname}_RMSE": lesst_rmse,
                        }
                    )
                except Exception as e:
                    print(
                        f"ERROR {dataset}_{n_cluster}_{localmodelname}_{globalmodelname}_ds:{deseason}: {e}"
                    )
                    continue

        try:
            total_lesst_owas.update({f"ds:{dataset}_OWA": lesst_owas})
            total_lesst_owas.update({f"ds:{dataset}_SMAPE": lesst_smapes})
            total_lesst_owas.update({f"ds:{dataset}_MASE": lesst_mases})
            total_lesst_owas.update({f"ds:{dataset}_RMSE": lesst_rmses})
            with open(
                f"E:/documents/work/thesis/lesst_{dataset}_ds_{deseason}.pkl",
                "wb",
            ) as handle:
                pickle.dump(
                    total_lesst_owas, handle, protocol=pickle.HIGHEST_PROTOCOL
                )
            if check_benchmark:
                t = time()
                (
                    benchmark_owa,
                    benchmark_smape,
                    benchmark_mase,
                    benchmark_rmse,
                ) = benchmark(
                    predictions,
                    dataset,
                    res_train,
                    res_test,
                    horizon,
                    frequency,
                )
                # benchmark_owa = 1
                print(f"Benchmark performance calculation time {time()-t} sec")

                total_benchmark_owas.update(
                    {f"ds:{dataset}_OWA": benchmark_owa}
                )
                total_benchmark_owas.update(
                    {f"ds:{dataset}_SMAPE": benchmark_smape}
                )
                total_benchmark_owas.update(
                    {f"ds:{dataset}_MASE": benchmark_mase}
                )
                total_benchmark_owas.update(
                    {f"ds:{dataset}_RMSE": benchmark_rmse}
                )
                with open(
                    f"E:/documents/work/thesis/benchmark_{dataset}_ds_{deseason}.pkl",
                    "wb",
                ) as handle:
                    pickle.dump(
                        total_benchmark_owas,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
        except Exception as e:
            print(
                f"ERROR {dataset}_{n_cluster}_{localmodelname}_{globalmodelname}_ds:{deseason}: {e}"
            )
            continue

    return total_benchmark_owas, total_lesst_owas
