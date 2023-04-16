# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 13:43:01 2022

@author: steph
"""
import numpy as np
import pandas as pd
from tsforecast import SeasonalNaive, ThetaF
from seasonality import Naive2
from sklearn.metrics import mean_squared_error


class PerformanceMeasures:
    def __init__(self, freq):
        self.freq = freq

    def sMAPE(self, real, predictions):
        horizon = len(predictions)
        smape = (
            2
            / horizon
            * np.sum(
                np.abs(real - predictions)
                / (np.abs(real) + np.abs(predictions))
            )
            * 100
        )
        return smape

    def MASE(self, real, predictions, train):
        horizon = len(predictions)
        n = len(train)
        m = self.freq
        mase_up = 1 / horizon * np.sum(np.abs(real - predictions))
        mase_down = 1 / (n - m) * np.sum(np.abs(train[m:] - train[:-m]))
        mase = mase_up / mase_down
        return mase

    def RMSE(self, real, predictions):
        rmse = mean_squared_error(real, predictions, squared=False)
        return rmse

    def OWA(self, real, predictions, train):
        model_owa = []
        model_smape = []
        model_mase = []
        model_rmse = []
        naive_smape = []
        naive_mase = []

        for i in range(0, len(train)):
            horizon = len(predictions[i])
            naive = Naive2(self.freq)
            naive.fit(train[i])
            naivepred = naive.predict(horizon)
            naive_smape.append(self.sMAPE(real[i], naivepred))
            naive_mase.append(self.MASE(real[i], naivepred, train[i]))
            model_smape.append(self.sMAPE(real[i], predictions[i]))
            model_mase.append(self.MASE(real[i], predictions[i], train[i]))
            model_rmse.append(self.RMSE(real[i], predictions[i]))
        measures = []
        for measure in [
            model_mase,
            model_smape,
            naive_mase,
            naive_smape,
            model_rmse,
        ]:
            measure = np.array(measure)
            measure[measure >= 1e308] = np.nan
            measures.append(np.nan_to_num(measure).mean())
        [
            model_mase,
            model_smape,
            naive_mase,
            naive_smape,
            model_rmse,
        ] = measures
        model_owa = (model_smape / naive_smape + model_mase / naive_mase) / 2
        return model_owa, model_smape, model_mase, model_rmse


class BenchmarkModel:
    def __init__(self, model):
        self.model = model

    def fit(self, train):
        self.model.fit(train)

    def predict(self, horizon):
        predictions = self.model.predict(horizon)
        return predictions

    def performance(self, real, train, horizon, freq):
        predictions = []
        for i in range(0, len(train)):
            self.fit(train[i])
            predictions.append(self.predict(horizon))
        measure = PerformanceMeasures(freq)
        owa, smape, mase, rmse = measure.OWA(real, predictions, train)
        return owa, smape, mase, rmse
