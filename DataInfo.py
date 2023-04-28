"""This file generates information about the dataset for each set"""
from data_prep import to_array
from preprocessing import read_m4test_series, read_m4_series
import numpy as np

np.seed = 1

datasets = [
    "Yearly",
    "Quarterly",
    "Monthly",
    "Weekly",
    "Daily",
    "Hourly",
]
total_length_timeseries = {}
avg_length_timeseries = {}
test_size = {}
for dataset in datasets:
    test = read_m4test_series(dataset)
    train = read_m4_series(dataset)
    res_train = to_array(train)
    res_test = to_array(test)

    # Determine the forecasting horizon for each dataset
    test_size.update({dataset: test.shape[1]})
    count = 0
    for i in res_train:
        count += len(i)

    # Determine the total series length for the dataset
    total_length_timeseries.update({dataset: count})

    # Determine average series length for the dataset
    avg_length_timeseries.update({dataset: count / len(res_test)})

    print("=====================================================")
    print(f"{dataset} is DONE")
    print("=====================================================")
