"""This file contains the Local and Global Model method"""
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from data_prep import last_values, rolling_window, rolling_output


class LocalModel:
    def __init__(self, modeltype=XGBRegressor(tree_method="gpu_hist")):
        """Define model to be used in the Local Models"""
        self.modeltype = modeltype
        self.models = {}

    def fit(self, X, Y):
        """Fit Local Models for each cluster"""
        for cluster in X.keys():
            model = MultiXgboost(self.modeltype)
            x = X[cluster]
            y = Y[cluster]
            model.fit(x, y)
            self.models.update({cluster: model})

    def predict(self, y, cluster):
        """Predict using the Local Model of specific cluster"""
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
        """Initiate Global Model"""
        self.model = model
        self.testsize = testsize
        self.localmodel = localmodel
        self.clusterids = clusterids
        self.local_weights = localweights

    def localpredictions(self, x, rolling=False):
        """Make predictions using the Local Models"""

        # rolling is used to use all data in the train set
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
        """Multiply predictions by their weights"""
        if not rolling:
            if not evenweight:
                preds = self.local_weights * preds
            else:
                # using equal weights instead of cluster weights
                preds = (
                    np.ones(self.local_weights.shape)
                    / self.local_weights.shape[1]
                    * preds
                )
        else:
            # Need to extend weight matrix for more datapoints
            weights = np.repeat(self.local_weights, repeats, axis=0)
            preds = weights * preds
            del weights
            del repeats
        return preds

    def fit(self, x, y, rolling=False, evenweight=False):
        """Fit the Global Model"""
        if rolling:
            repeats = np.repeat(
                [(len(i) - self.testsize) for i in x], self.testsize
            )
            y = rolling_output(x, self.testsize)
        else:
            repeats = []
        # Make predictions using Local Models
        preds = self.localpredictions(x, rolling)

        # Weight the predictions
        x = self.weightedpredictions(preds, rolling, repeats, evenweight)
        y = np.array(y).reshape(-1, 1)

        # Fit the model
        self.model.fit(x, y)
        del x
        del y

    def predict(self, values, evenweight=False):
        """Make predictions using the Global Model"""
        x = np.array(last_values(values, timesteps=self.testsize))
        x = self.localpredictions(x)
        x = self.weightedpredictions(x, evenweight=evenweight)
        return self.model.predict(x)


class WeightedSum:
    def __init__(self):
        """Initiate weighted sum method"""
        pass

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, values):
        """Predicts the Global Model using the sum of weighted forecasts"""
        return values.sum(axis=1)


class MultiXgboost(MultiOutputRegressor):
    def __init__(self, model=XGBRegressor(tree_method="gpu_hist")):
        """Method that creates and fits multiple models for each timestep
        to be predicted"""
        self.estimator = model
        self.n_jobs = None
