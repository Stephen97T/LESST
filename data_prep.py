import numpy as np

"""
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
"""


def split_train_val_test(data, testsize):
    ytrain = []
    yval = []
    ytest = []
    for ts in data.index:
        y = np.array(data.loc[ts])
        y = y[~np.isnan(y)]
        ytrain.append(y[: -2 * testsize])
        yval.append(y[-2 * testsize : -testsize])
        ytest.append(y[-testsize:])
    return ytrain, yval, ytest


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


def prepare_inputoutput(df, testsize):
    inputs = {}
    outputs = {}
    for i in df.cluster.unique():
        data = df[df.cluster == i]
        data = data.drop("cluster", axis=1)
        train, val, test = split_train_val_test(data, testsize)
        X, Y = prepare_train(ytrain=train, timesteps=testsize)
        inputs.update({i: X})
        outputs.update({i: Y})
    return inputs, outputs
