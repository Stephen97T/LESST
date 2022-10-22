#!/usr/bin/env python
# coding: utf-8
import os
from time import time

os.environ["R_HOME"] = "/Users/steph/miniconda3/envs/bay/lib/R"
from typing import List
from tsfeatures.tsfeatures_r import tsfeatures_r
import pandas as pd
from rpy2.robjects.packages import importr, data

# utils = importr("utils")
# utils.install_packages("data.table")
# utils.install_packages("tsfeatures")
from tsfeatures.tsfeatures_r import tsfeatures_r
from tsfeatures.tsfeatures import tsfeatures
from tsfeatures.m4_data import *
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# xtrain, ytrain, xtest, ytest = m4_parser("Monthly", "C:/thesis/data/")
# panel = ytrain[ytrain.unique_id == "M1"]
"""
t = time()
b = tsfeatures(panel, freq=1)
print(f"python finished in {time()-t} sec")
t = time()
a = tsfeatures_r(panel, freq=1)
with localconverter(ro.default_converter + pandas2ri.converter):
    c = ro.conversion.rpy2py(a)
print(f"R finished in {time()-t} sec")
"""
