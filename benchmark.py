"""
This file contains the functions neccesary for calculating 
the performance measures and the benchmarks
"""
import numpy as np
from seasonality import Naive2
from sklearn.metrics import mean_squared_error


class PerformanceMeasures:
    def __init__(self, freq):
        """Set frequency for PerformanceMeasure object"""
        self.freq = freq

    def sMAPE(self, real, predictions):
        """Calculate sMape for one series"""
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
        """Calculate MASE for one series"""
        horizon = len(predictions)
        n = len(train)
        m = self.freq
        mase_up = 1 / horizon * np.sum(np.abs(real - predictions))
        mase_down = 1 / (n - m) * np.sum(np.abs(train[m:] - train[:-m]))
        mase = mase_up / mase_down
        return mase

    def RMSE(self, real, predictions):
        """Calculate RMSE for one series"""
        rmse = mean_squared_error(real, predictions, squared=False)
        return rmse

    def OWA(self, real, predictions, train):
        """Calculate the OWA for all series",
        in this case real and predictions
        contain a list of values/predictions
        for multiple series"""
        model_owa = []
        model_smape = []
        model_mase = []
        model_rmse = []
        naive_smape = []
        naive_mase = []

        # Calculate performance measures for each series
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

        # Take the Mean for over all series for all measures
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
        """Set the model to be used as benchmark model"""
        self.model = model

    def fit(self, train):
        """Fit on one time series"""
        self.model.fit(train)

    def predict(self, horizon):
        """Make predictions for one series"""
        predictions = self.model.predict(horizon)
        return predictions

    def performance(self, real, train, horizon, freq):
        """Calculate performance measures using the benchmark model
        for a whole dataset, train consists of a list of
        timeseries arrays"""
        predictions = []
        for i in range(0, len(train)):
            self.fit(train[i])
            predictions.append(self.predict(horizon))
        measure = PerformanceMeasures(freq)
        owa, smape, mase, rmse = measure.OWA(real, predictions, train)
        return owa, smape, mase, rmse
