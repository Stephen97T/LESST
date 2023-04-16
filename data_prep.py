import numpy as np
from seasonality import Seasonality


def split_train_val(data, testsize, freq, deseason=True, split=True):
    ytrain = []
    yval = []
    seasonalities = []
    for ts in data.index:
        y = np.array(data.loc[ts])
        y = y[~np.isnan(y)]
        if deseason:
            seas = Seasonality(freq)
            ytr = seas.deseasonalize_serie(y)
            seasonalities.append(seas)
        else:
            ytr = y
        if split:
            ytrain.append(ytr[:-testsize])
        else:
            ytrain.append(ytr)

        yval.append(ytr[-testsize:])

    return ytrain, yval, seasonalities


def to_array(data):
    series = []
    for ts in data.index:
        serie = np.array(data.loc[ts])
        serie = serie[~np.isnan(serie)]
        series.append(serie)
    return series


def shift(data, timesteps, start=True):
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
    last = []
    for data in ytrain:
        last.append([data[-1]] * timesteps)
    return last


def rolling_window(ytrain, timesteps):
    last = []
    for data in ytrain:
        for dat in data[:-timesteps]:
            last.append([dat] * timesteps)
    return last


def rolling_output(youtput, timesteps):
    last = []
    for data in youtput:
        for dat in range(0, (len(data) - timesteps)):
            last.append(data[dat : timesteps + dat])
    return last


def prepare_inputoutput(
    df, testsize, freq, deseason=True, split=True, start=True
):
    inputs = {}
    outputs = {}
    for i in df.cluster.unique():
        data = df[df.cluster == i]
        data = data.drop("cluster", axis=1)
        train, val, _ = split_train_val(data, testsize, freq, deseason, split)
        X, Y = prepare_train(ytrain=train, timesteps=testsize, start=start)
        inputs.update({i: X})
        outputs.update({i: Y})
    return inputs, outputs
