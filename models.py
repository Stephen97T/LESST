import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from data_prep import last_values, rolling_window


class LocalModel:
    def __init__(self, modeltype=XGBRegressor(tree_method="gpu_hist")):
        self.modeltype = modeltype
        self.models = {}

    def fit(self, X, Y):
        for cluster in X.keys():
            model = MultiXgboost(self.modeltype)
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
        model=XGBRegressor(tree_method="gpu_hist"),
    ):
        self.model = model
        self.testsize = testsize
        self.localmodel = localmodel
        self.clusterids = clusterids
        self.local_weights = localweights

    def localpredictions(self, x, rolling=False):
        if rolling:
            x = np.array(rolling_window(x, timesteps=self.testsize))
        else:
            x = np.array(last_values(x, timesteps=self.testsize))

        preds = []
        for i in range(0, self.clusterids.shape[1]):
            # predict for all series the predictions for each local model
            preds.append(
                self.localmodel.predict(x, cluster=i).reshape(
                    -1,
                )
            )
        preds = np.transpose(np.array(preds))
        return preds

    def weightedpredictions(self, preds):
        totalpred = self.local_weights * preds
        return totalpred

    def fit(self, x, y, rolling=False):
        preds = self.localpredictions(x, rolling)
        totalpred = self.weightedpredictions(preds)
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
    def __init__(self, model=XGBRegressor(tree_method="gpu_hist")):
        self.estimator = model
        self.n_jobs = None
