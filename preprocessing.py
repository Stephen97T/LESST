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
import pandas as pd

# import os
# os.environ["R_HOME"] = "/Users/steph/miniconda3/envs/bay/lib/R"
from tsfeatures.m4_data import *


def prepare_allm4data(
    outputfolder="C:/thesis/data/m4/processed", datapath="C:/thesis/data/"
):
    datasets = ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
    for dset in datasets:
        xtrain, ytrain, xtest, ytest = m4_parser(dset, datapath)
        xtrain.to_csv(f"{outputfolder}/train/xtrain-{dset.lower()}.csv")
        ytrain.to_csv(f"{outputfolder}/train/ytrain-{dset.lower()}.csv")
        xtest.to_csv(f"{outputfolder}/test/xtest-{dset.lower()}.csv")
        ytest.to_csv(f"{outputfolder}/test/ytest-{dset.lower()}.csv")


def prepare_m4tsfeatures(
    filename="features_train_val",
    outputfolder="C:/thesis/data/m4/processed/tsfeatures",
    datapath="C:/thesis/data/m4/processed/train",
):
    datasets = ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
    for dset in datasets:
        y = pd.read_csv(f"{datapath}/ytrain-{dset.lower()}.csv", index_col=0)
        features = tsfeatures_r(y, freq=1)
        with localconverter(ro.default_converter + pandas2ri.converter):
            features = ro.conversion.rpy2py(features)
        features.to_csv(f"{outputfolder}/{filename}-{dset.lower()}.csv")


def read_m4_df(
    dataset, train_test="train", datapath="C:/thesis/data/m4/processed/"
):
    df = pd.read_csv(
        f"{datapath}/{train_test}/ytrain-{dataset}.csv", index_col=0
    )
    return df


def read_m4_series(dataset, train_test="Train", datapath="C:/thesis/data/m4"):
    df = pd.read_csv(
        f"{datapath}/{train_test}/{dataset}-{train_test.lower()}.csv",
        index_col=0,
    )
    return df


def read_tsfeatures(dataset, datapath="C:/thesis/data/m4/processed/"):
    df = pd.read_csv(
        f"{datapath}/tsfeatures/features_train_val-{dataset}.csv", index_col=0
    )
    return df


def split_train_val_test(data, testsize):
    ytrain = []
    yval = []
    ytest = []
    for ts in data.unique_id.unique():
        y = np.array(data[data.unique_id == ts].y)
        ytrain.append(y[: -2 * testsize])
        yval.append(y[-2 * testsize : -testsize])
        ytest.append(y[-testsize:])
    return ytrain, yval, ytest
