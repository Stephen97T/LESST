"""This file contians functions for running the main results,
contains functions for:
-Running LESST Model for all datasets and multiple parameters
-Calculating performance measures for this model
-Calculating benchmark performance
-Calculating Local Model performance"""
from LESST import LESST
from benchmark import BenchmarkModel, PerformanceMeasures
from tsforecast import ThetaF
import numpy as np
from data_prep import to_array, last_values
from preprocessing import read_m4test_series, read_m4_series
from time import time
import pickle


def local_results(less, dataset, res_train, res_test, predictions, deseason):
    """Calculates the results for the local models"""
    local = less.LocalM
    clusters = less.df.cluster
    train = less.val
    localpred = []
    inputs = np.array(last_values(train, timesteps=less.steps))
    for i, y in enumerate(inputs):
        cluster = clusters[i]
        predic = local.predict(y.reshape(1, -1), cluster=cluster)
        localpred.append(predic)
    localpred = np.array(localpred).reshape(predictions.shape)
    if deseason:
        preds = []
        for i in range(0, len(less.seas)):
            preds.append(less.seas[i].reseasonalize_pred(localpred[i]))
        localpred = np.array(preds)

    local_owa, local_smape, local_mase, local_rmse, = performance_LESST(
        localpred,
        dataset,
        res_train,
        res_test,
        less.steps,
        less.freq,
    )
    return local_owa, local_smape, local_mase, local_rmse


def run_LESST(
    dataset,
    train,
    n_clusters,
    horizon,
    frequency,
    localmodel,
    globalmodel,
    deseason,
    rolling=False,
    evenweighted=False,
):
    """Runs and makes predictions for the LESST Model"""
    less = LESST(
        dataset,
        train,
        n_clusters,
        frequency,
        deseason,
        split=True,
        start=True,
        rolling=rolling,
        evenweight=evenweighted,
    )
    less.fit(
        prediction_steps=horizon,
        localmodel=localmodel,
        globalmodel=globalmodel,
    )

    predictions = less.predict()
    return predictions, less


def performance_LESST(predictions, dataset, train, test, horizon, frequency):
    """Calculates the performance of the LESST model"""
    measure = PerformanceMeasures(frequency)
    lesst_owa, lesst_smape, lesst_mase, lesst_rmse = measure.OWA(
        test, predictions, train
    )
    return lesst_owa, lesst_smape, lesst_mase, lesst_rmse


def benchmark(predictions, dataset, train, test, horizon, frequency):
    """Calculates benchmark performance using the Theta model"""
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
    """Creates our main results, muliple LESST models can be trained and evaluated
    with this function"""
    total_lesst_owas = {}
    total_benchmark_owas = {}
    for instance, dataset in enumerate(datasets):

        # Initiate dataset
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

                    # Fit and predict with LESST model
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

                    # Calculate performance
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

            # Save Performance Measure results
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

            # Calculate benchmark performance
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
