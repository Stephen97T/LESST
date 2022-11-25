import numpy as np


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
    X = []
    Y = []
    for data in ytrain:
        x, y = shift(data, timesteps)
        X.append(x)
        Y.append(y)
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    return X, Y


def last_values(ytrain):
    last = []
    for data in ytrain:
        last.append(data[-1])
    return last
