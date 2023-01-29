# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 13:43:01 2022

@author: steph
"""
import numpy as np
import pandas as pd
from tsforecast import SeasonalNaive, ThetaF
from seasonality import Naive2


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

    def OWA(self, real, predictions, train):
        horizon = len(predictions)
        naive = Naive2(self.freq)
        naive.fit(train)
        naivepred = naive.predict(horizon)
        naive_smape = self.sMAPE(real, naivepred)
        naive_mase = self.MASE(real, naivepred, train)
        model_smape = self.sMAPE(real, predictions)
        model_mase = self.MASE(real, predictions, train)
        owa = (model_smape / naive_smape + model_mase / naive_mase) / 2
        return owa


class BenchmarkModel:
    def __init__(self, model):
        self.model = model

    def fit(self, train):
        self.model.fit(train)

    def predict(self, horizon):
        predictions = self.model.predict(horizon)
        return predictions

    def performance(self, real, train, horizon, freq):
        self.fit(train)
        predictions = self.predict(horizon)
        measure = PerformanceMeasures(freq)
        owa = measure.OWA(real, predictions, train)
        return owa
