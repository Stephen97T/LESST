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
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense


class RNN:
    def __init__(self, dim_input, dim_output):
        self.model = self.initiate_model(dim_input, dim_output)

    def initiate_model(self, dim_input, dim_output):
        model = Sequential()
        # Embedding layer
        model.add(
            Embedding(
                input_dim=dim_input,
                output_dim=dim_output,
            )
        )
        model.add(
            LSTM(
                64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1
            )
        )

        # Fully connected layer
        model.add(Dense(64, activation="relu"))

        # Dropout for regularization
        # model.add(Dropout(0.5))

        # Output layer
        model.add(Dense(dim_input, activation="sigmoid"))
        return model


class LocalModel:
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
