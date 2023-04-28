"""File contains functions for formatting and reading m4data and timeseries features"""
import os
from time import time

# VERY IMPORTANT: Set you R module for the conda environment (work is the environment here)
os.environ["R_HOME"] = "E:/documents/work/mini/envs/work/lib/R"
from typing import List
from tsfeatures.tsfeatures_r import tsfeatures_r
import pandas as pd
from rpy2.robjects.packages import importr, data

"""MUST INSTALL these R libraries for running the first time"""
# utils = importr("utils")
# utils.install_packages("data.table")
# utils.install_packages("tsfeatures")
# utils.install_packages("forecast")
from tsfeatures.tsfeatures_r import tsfeatures_r
from tsfeatures.tsfeatures import tsfeatures
from tsfeatures.m4_data import *
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import pandas as pd

# import os
# os.environ["R_HOME"] = "/Users/steph/miniconda3/envs/bay/lib/R"
from tsfeatures.m4_data import *


def prepare_allm4data(
    outputfolder="E:/documents/work/thesis/data/m4/processed",
    datapath="E:/documents/work/thesis/data",
):
    """Prepares m4 data for calculating the features"""
    datasets = ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
    for dset in datasets:
        xtrain, ytrain, xtest, ytest = m4_parser(dset, datapath)
        xtrain.to_csv(f"{outputfolder}/train/xtrain-{dset.lower()}.csv")
        ytrain.to_csv(f"{outputfolder}/train/ytrain-{dset.lower()}.csv")
        xtest.to_csv(f"{outputfolder}/test/xtest-{dset.lower()}.csv")
        ytest.to_csv(f"{outputfolder}/test/ytest-{dset.lower()}.csv")


def prepare_m4tsfeatures(
    filename="features_train_val",
    outputfolder="E:/documents/work/thesis/data/m4/processed/tsfeatures",
    datapath="E:/documents/work/thesis/data/m4/processed/train",
):
    """Calculates the features for all datasets and saves them"""
    datasets = ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
    for dset in datasets:
        y = pd.read_csv(f"{datapath}/ytrain-{dset.lower()}.csv", index_col=0)
        features = tsfeatures_r(y, freq=1)
        with localconverter(ro.default_converter + pandas2ri.converter):
            features = ro.conversion.rpy2py(features)
        features.to_csv(f"{outputfolder}/{filename}-{dset.lower()}.csv")


def read_m4_df(
    dataset,
    train_test="train",
    datapath="E:/documents/work/thesis/data/m4/processed/",
):
    """Reads m4 transformed data (used for timeseries features)"""
    df = pd.read_csv(
        f"{datapath}/{train_test}/ytrain-{dataset}.csv", index_col=0
    )
    return df


def read_m4_series(
    dataset, train_test="Train", datapath="E:/documents/work/thesis/data/m4"
):
    """Reads the normal m4 data (much faster)"""
    df = pd.read_csv(
        f"{datapath}/{train_test}/{dataset}-{train_test.lower()}.csv",
        index_col=0,
    )
    return df


def read_m4test_series(
    dataset, train_test="Test", datapath="E:/documents/work/thesis/data/m4"
):
    """Reads the normal m4 test data"""
    df = pd.read_csv(
        f"{datapath}/{train_test}/{dataset}-{train_test.lower()}.csv",
        index_col=0,
    )
    return df


def read_tsfeatures(
    dataset, datapath="E:/documents/work/thesis/data/m4/processed/"
):
    """Reads the saved timeseries features previously calculated"""
    df = pd.read_csv(
        f"{datapath}/tsfeatures/features_train_val-{dataset}.csv", index_col=0
    )
    return df
