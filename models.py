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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from data_prep import last_values


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
    def __init__(self):
        self.models = {}

    def fit(self, X, Y):
        for cluster in X.keys():
            model = MultiXgboost()
            x = X[cluster]
            y = Y[cluster]
            model.fit(x, y)
            self.models.update({cluster: model})

    def predict(self, y, cluster):
        pred = self.models[cluster].predict(y)
        return pred


class GlobalModel:
    def __init__(
        self,
        localmodel,
        testsize,
        clusterids,
        localweights,
        model=RandomForestRegressor(),
    ):
        self.model = model
        self.testsize = testsize
        self.localmodel = localmodel
        self.clusterids = clusterids
        self.local_weights = localweights

    def localpredictions(self, x):
        x = np.array(last_values(x, timesteps=self.testsize))

        preds = {}
        for i in range(0, self.clusterids.shape[1]):
            # predict for all series the predictions for each local model
            preds.update({i: self.localmodel.predict(x, cluster=i)})
        return preds

    def weightedpredictions(self, preds):
        totalpred = []
        for i in range(0, self.clusterids.shape[0]):
            ypred = []
            for j in range(0, self.clusterids.shape[1]):
                # multiply local predictions with the cluster weights
                ypred.append(
                    self.local_weights[i, j] * preds[self.clusterids[i, j]][i]
                )
            # save all weighted predictions
            totalpred.append(np.array(ypred).T)
        totalpred = np.array(totalpred)
        return totalpred

    def fit(self, x, y):
        preds = self.localpredictions(x)
        totalpred = self.weightedpredictions(preds)
        # reshape data for training the global model
        x = totalpred.reshape(
            totalpred.shape[0] * totalpred.shape[1], totalpred.shape[2]
        )
        y = np.array(y).reshape(-1, 1)
        self.model.fit(x, y)

    def predict(self, values):
        x = np.array(last_values(values, timesteps=self.testsize))

        preds = self.localpredictions(x)
        totalpred = self.weightedpredictions(preds)

        x = totalpred.reshape(
            totalpred.shape[0] * totalpred.shape[1], totalpred.shape[2]
        )
        return self.model.predict(x)


class WeightedSum:
    def __init__(self):
        pass

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, values):
        return self.x.sum(axis=1)


class MultiXgboost(MultiOutputRegressor):
    def __init__(self):
        self.estimator = XGBRegressor()
        self.n_jobs = None

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
