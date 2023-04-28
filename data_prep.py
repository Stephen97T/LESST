"""This file contains functions for preparing the data to fit the 
Local and Global model"""
import numpy as np
from seasonality import Seasonality


def split_train_val(data, testsize, freq, deseason=True, split=True):
    """Split train set into a train and validation set using the testsize
    if deseason True then use deseasonalization
    if split = True then remove the validation
    from the train set"""
    ytrain = []
    yval = []
    seasonalities = []
    for ts in data.index:
        y = np.array(data.loc[ts])
        y = y[~np.isnan(y)]
        if deseason:
            # Deseasonalize
            seas = Seasonality(freq)
            ytr = seas.deseasonalize_serie(y)

            # Save seasonalities for each series
            seasonalities.append(seas)
        else:
            ytr = y
        if split:
            # Split validation from train set
            ytrain.append(ytr[:-testsize])
        else:
            ytrain.append(ytr)

        yval.append(ytr[-testsize:])

    return ytrain, yval, seasonalities


def to_array(data):
    """Makes whole dataset into a list of arrays"""
    series = []
    for ts in data.index:
        serie = np.array(data.loc[ts])
        serie = serie[~np.isnan(serie)]
        series.append(serie)
    return series


def shift(data, timesteps, start=True):
    """Shifts the input and target for calculating the h steps ahead
    if start is True then input starts at t=1"""
    ones = np.ones(timesteps).reshape(-1, 1).T
    if start:
        x = np.dot(data[:-timesteps].reshape(-1, 1), ones)
        y = np.array(
            [
                data[i + 1 : (i - (timesteps - 1))]
                if i < (timesteps - 1)
                else data[i + 1 :]
                for i in range(0, timesteps)
            ]
        ).T
    else:
        y = np.dot(data[timesteps:].reshape(-1, 1), ones)
        x = np.array(
            [data[timesteps - (i + 1) : -(i + 1)] for i in range(0, timesteps)]
        ).T
    return x, y


def prepare_train(ytrain, timesteps, start=True):
    """Prepare input and target for Local Models for
    one cluster"""
    # ytrain is the training data per cluster
    X = []
    Y = []
    for data in ytrain:
        x, y = shift(data, timesteps, start)
        X.append(x)
        Y.append(y)
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    return X, Y


def last_values(ytrain, timesteps):
    """Return last values for multiple series"""
    last = []
    for data in ytrain:
        last.append([data[-1]] * timesteps)
    return last


def rolling_window(ytrain, timesteps):
    """Make a rolling window for the input set using multiple series
    with the length of the timesteps"""
    last = []
    for data in ytrain:
        for dat in data[:-timesteps]:
            last.append([dat] * timesteps)
    return last


def rolling_output(youtput, timesteps):
    """Make a rolling window for the target set"""
    last = []
    for data in youtput:
        for dat in range(0, (len(data) - timesteps)):
            last.append(data[dat : timesteps + dat])
    return last


def prepare_inputoutput(
    df, testsize, freq, deseason=True, split=True, start=True
):
    """Prepare input and target for the Local Model
    and separate them per cluster"""
    inputs = {}
    outputs = {}
    for i in df.cluster.unique():

        # Select data only from the cluster
        data = df[df.cluster == i]
        data = data.drop("cluster", axis=1)

        # Split series
        train, val, _ = split_train_val(data, testsize, freq, deseason, split)

        # Prepare input and target
        X, Y = prepare_train(ytrain=train, timesteps=testsize, start=start)

        # Save values
        inputs.update({i: X})
        outputs.update({i: Y})
    return inputs, outputs
