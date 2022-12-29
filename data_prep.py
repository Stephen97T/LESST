import numpy as np
from seasonality import Seasonality


def split_train_val_test(data, testsize, freq):
    ytrain = []
    yval = []
    ytest = []
    seasonalities = []
    for ts in data.index:
        seas = Seasonality(freq)
        y = np.array(data.loc[ts])
        y = y[~np.isnan(y)]
        ytr = seas.deseasonalize_serie(y[:-testsize])
        ytrain.append(ytr[:-testsize])
        yval.append(ytr[-testsize:])
        ytest.append(y[-testsize:])
        seasonalities.append(seas)
    return ytrain, yval, ytest, seasonalities


def shift(data, timesteps):
    ones = np.ones(timesteps).reshape(-1, 1).T
    x = np.dot(data[:-timesteps].reshape(-1, 1), ones)
    y = np.array(
        [
            data[i + 1 : (i - (timesteps - 1))]
            if i < (timesteps - 1)
            else data[i + 1 :]
            for i in range(0, timesteps)
        ]
    ).T
    return x, y


def prepare_train(ytrain, timesteps):
    # ytrain is the training data per cluster
    X = []
    Y = []
    for data in ytrain:
        x, y = shift(data, timesteps)
        X.append(x)
        Y.append(y)
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    return X, Y


def last_values(ytrain, timesteps):
    last = []
    for data in ytrain:
        last.append([data[-1]] * timesteps)
    return last


def prepare_inputoutput(df, testsize, freq):
    inputs = {}
    outputs = {}
    for i in df.cluster.unique():
        data = df[df.cluster == i]
        data = data.drop("cluster", axis=1)
        train, val, test, _ = split_train_val_test(data, testsize, freq)
        X, Y = prepare_train(ytrain=train, timesteps=testsize)
        inputs.update({i: X})
        outputs.update({i: Y})
    return inputs, outputs
