import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from data_prep import last_values, rolling_window, rolling_output


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

    def weightedpredictions(
        self, preds, rolling=False, repeats=[], evenweight=False
    ):
        if not rolling:
            if not evenweight:
                totalpred = self.local_weights * preds
            else:
                totalpred = (
                    np.ones(self.local_weights.shape)
                    / self.local_weights.shape[1]
                    * preds
                )
        else:
            weights = np.repeat(self.local_weights, repeats, axis=0)
            totalpred = weights * preds
        return totalpred

    def fit(self, x, y, rolling=False, evenweight=False):
        if rolling:
            repeats = np.repeat(
                [(len(i) - self.testsize) for i in x], self.testsize
            )
            y = rolling_output(x, self.testsize)
        else:
            repeats = []
        preds = self.localpredictions(x, rolling)
        x = self.weightedpredictions(preds, rolling, repeats, evenweight)
        y = np.array(y).reshape(-1, 1)
        self.model.fit(x, y)

    def predict(self, values, evenweight=False):
        x = np.array(last_values(values, timesteps=self.testsize))
        x = self.localpredictions(x)
        x = self.weightedpredictions(x, evenweight=evenweight)
        return self.model.predict(x)


class WeightedSum:
    def __init__(self):
        pass

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, values):
        return values.sum(axis=1)


class MultiXgboost(MultiOutputRegressor):
    def __init__(self, model=XGBRegressor(tree_method="gpu_hist")):
        self.estimator = model
        self.n_jobs = None
