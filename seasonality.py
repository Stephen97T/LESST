import numpy as np
from numpy.random import seed

seed(1)

import pandas as pd
from math import sqrt


########################
# UTILITY MODELS
########################


def detrend(insample_data):
    """
    Calculates a & b parameters of LRL
    :param insample_data:
    :return:
    """
    x = np.arange(len(insample_data))
    a, b = np.polyfit(x, insample_data, 1)
    return a, b


def deseasonalize(original_ts, ppy):
    """
    Calculates and returns seasonal indices
    :param original_ts: original data
    :param ppy: periods per year
    :return:
    """
    """
  # === get in-sample data
  original_ts = original_ts[:-out_of_sample]
  """
    if seasonality_test(original_ts, ppy):
        # ==== get moving averages
        ma_ts = moving_averages(original_ts, ppy)

        # ==== get seasonality indices
        le_ts = original_ts * 100 / ma_ts
        le_ts = np.hstack((le_ts, np.full((ppy - (len(le_ts) % ppy)), np.nan)))
        le_ts = np.reshape(le_ts, (-1, ppy))
        si = np.nanmean(le_ts, 0)
        norm = np.sum(si) / (ppy * 100)
        si = si / norm
    else:
        si = np.ones(ppy)

    return si


def moving_averages(ts_init, window):
    """
    Calculates the moving averages for a given TS
    :param ts_init: the original time series
    :param window: window length
    :return: moving averages ts
    """
    """
  As noted by Professor Isidro Lloret Galiana:
  line 82:
  if len(ts_init) % 2 == 0:
  
  should be changed to
  if window % 2 == 0:
  
  This change has a minor (less then 0.05%) impact on the calculations of the seasonal indices
  In order for the results to be fully replicable this change is not incorporated into the code below
  """
    ts_init = pd.Series(ts_init)

    if len(ts_init) % 2 == 0:
        ts_ma = ts_init.rolling(window, center=True).mean()
        ts_ma = ts_ma.rolling(2, center=True).mean()
        ts_ma = np.roll(ts_ma, -1)
    else:
        ts_ma = ts_init.rolling(window, center=True).mean()

    return ts_ma


def seasonality_test(original_ts, ppy):
    """
    Seasonality test
    :param original_ts: time series
    :param ppy: periods per year
    :return: boolean value: whether the TS is seasonal
    """
    s = acf(original_ts, 1)
    for i in range(2, ppy):
        s = s + (acf(original_ts, i) ** 2)

    limit = 1.645 * (sqrt((1 + 2 * s) / len(original_ts)))

    return (abs(acf(original_ts, ppy))) > limit


def acf(data, k):
    """
    Autocorrelation function
    :param data: time series
    :param k: lag
    :return:
    """
    m = np.mean(data)
    s1 = 0
    for i in range(k, len(data)):
        s1 = s1 + ((data[i] - m) * (data[i - k] - m))

    s2 = 0
    for i in range(0, len(data)):
        s2 = s2 + ((data[i] - m) ** 2)

    return float(s1 / s2)


class Naive:
    """
    Naive model.
    This benchmark model produces a forecast that is equal to
    the last observed value for a given time series.
    """

    def __init__(self):
        pass

    def fit(self, ts_init):
        """
        ts_init: the original time series
        ts_naive: last observations of time series
        """
        self.ts_naive = [ts_init[-1]]
        return self

    def predict(self, h):
        return np.array(self.ts_naive * h)


class SeasonalNaive:
    """
    Seasonal Naive model.
    This benchmark model produces a forecast that is equal to
    the last observed value of the same season for a given time
    series.
    """

    def __init__(self):
        pass

    def fit(self, ts_init, seasonality):
        """
        ts_init: the original time series
        frcy: frequency of the time series
        ts_naive: last observations of time series
        """
        self.ts_seasonal_naive = ts_init[-seasonality:]
        return self

    def predict(self, h):
        repetitions = int(np.ceil(h / len(self.ts_seasonal_naive)))
        y_hat = np.tile(self.ts_seasonal_naive, reps=repetitions)[:h]
        return y_hat


class Naive2:
    """
    Naive2 model.
    Popular benchmark model for time series forecasting that automatically adapts
    to the potential seasonality of a series based on an autocorrelation test.
    If the series is seasonal the model composes the predictions of Naive and SeasonalNaive,
    else the model predicts on the simple Naive.
    """

    def __init__(self, seasonality):
        self.seasonality = seasonality

    def fit(self, ts_init):
        seasonality_in = deseasonalize(ts_init, ppy=self.seasonality)
        windows = int(np.ceil(len(ts_init) / self.seasonality))

        self.ts_init = ts_init
        self.s_hat = np.tile(seasonality_in, reps=windows)[: len(ts_init)]
        self.ts_des = ts_init / self.s_hat

        return self

    def predict(self, h):
        s_hat = (
            SeasonalNaive()
            .fit(self.s_hat, seasonality=self.seasonality)
            .predict(h)
        )
        r_hat = Naive().fit(self.ts_des).predict(h)
        y_hat = s_hat * r_hat
        return y_hat


class Seasonality:
    def __init__(self, freq):
        self.freq = freq

    def deseasonalize_serie(self, serie):
        seasonality_in = deseasonalize(serie, ppy=self.freq)
        windows = int(np.ceil(len(serie) / self.freq))
        self.s_hat = np.tile(seasonality_in, reps=windows)[: len(serie)]
        ts_des = serie / self.s_hat
        return ts_des

    def reseasonalize_serie(self, serie):
        return serie * self.s_hat

    def reseasonalize_pred(self, pred):
        horizon = len(pred)
        s_hat = (
            SeasonalNaive()
            .fit(self.s_hat, seasonality=self.freq)
            .predict(horizon)
        )
        y_hat = s_hat * pred
        return y_hat
