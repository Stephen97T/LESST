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
