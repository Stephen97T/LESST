import pandas as pd
import numpy as np
from tsforecast import (
    ARIMA,
    Naive,
    SeasonalNaive,
    ETS,
    NNETAR,
    TBATS,
    STLM,
    RandomWalk,
    ThetaF,
)
import tensorflow as tf
from keras import Sequential, losses
from keras.optimizers import Adam
from keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor


class RNN:
    def __init__(self, dim_input, dim_output, n_obs):
        self.model = self.initiate_model(dim_input, dim_output, n_obs)

    def initiate_model(self, dim_input, dim_output, n_obs):
        model = Sequential()
        model.add(LSTM(128, input_shape=(n_obs, dim_input)))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(dim_output))
        model.compile(
            optimizer=Adam(0.0001), loss=losses.MeanAbsolutePercentageError()
        )

        # Fully connected layer
        # model.add(Dense(64, activation="relu"))
        # model.add(Dense(dim_output, activation="sigmoid"))
        # Dropout for regularization
        # model.add(Dropout(0.5))
        # model.compile(optimizer="adam", loss=losses.MeanAbsoluteError())
        return model

    def split_train_val_test(self):
        pass

    def reshape_inputdata(self, data, n_observations, n_features, n_samples=1):
        # n_observations = x.shape[0]
        # n_features = x.shape[1]
        return data.reshape((n_samples, n_observations, n_features))

    def reshape_outputdata(self, data, n_samples=1):
        return data.reshape((n_samples, len(data)))


class LocalModel:
    def __init__(self, model):
        self.model = model

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, y):
        pred = self.model.predict(y)
        return pred


class MultiXgboost(MultiOutputRegressor):
    def __init__(self):
        self.estimator = XGBRegressor()
        self.n_jobs = None


class GradientBoost:
    def __init__(self):
        pass

    """
    def __init__(self, modeltype, freq):
        self.modeltype = modeltype
        self.model = self.initiate_model(modeltype, freq)

    def initiate_model(self, modeltype, freq):
        if modeltype == "tsforecast":
            model = {
                "arima": ARIMA(freq),
                "naive": Naive(freq),
                "snaive": SeasonalNaive(freq),
                "ets": ETS(freq),
                "nnetar": NNETAR(freq),
                "tbats": TBATS(freq),
                "stlm": STLM(freq),
                "rw": RandomWalk(freq),
                "thetaf": ThetaF(freq),
            }
        return model

    def fit(self, y):
        for key in self.model.keys():
            self.model[key].fit(y)
        if self.modeltype == "tsforecast":
            predictions = self.ts_predict(1)
        return self

    def ts_predict(self, steps):
        predictions = []
        for key in self.model.keys():
            predictions.append(self.model[key].predict(steps))
        return np.array(predictions)
"""
