# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 13:43:01 2022

@author: steph
"""
import numpy as np
import pandas as pd


class PerformanceMeasures:
    def __init__(self, freq):
        self.freq = freq

    def sMAPE(self, real, predictions, horizon):
        smape = (
            2
            / horizon
            * np.sum(real - predictions / (np.abs(real) + np.abs(predictions)))
            * 100
        )
        return smape
