from preprocessing import *
from timeseries import *
from tsforecast import *
from models import *
from data_prep import *
from clustering import *
import numpy as np

df = read_m4_df("monthly")
feats = read_tsfeatures("monthly")

df = df[df.unique_id.isin(["M1", "M2", "M3", "M4"])].reset_index(drop=True)
feats = feats[feats.unique_id.isin(["M1", "M2", "M3", "M4"])].reset_index(
    drop=True
)

clust = FeatureClustering(3)
clust.cluster_features(feats, df)
local_weights = clust.cluster_distances
clusterids = clust.idcluster_distance

testsize = 12
ytrain = {}
yval = {}
ytest = {}
inputs = {}
outputs = {}
lastvalues = {}
for i in df.cluster.unique():
    data = df[df.cluster == i]
    train, val, test = split_train_val_test(data, testsize)
    X, Y = prepare_train(ytrain=train, timesteps=testsize)
    ytrain.update({i: train})
    yval.update({i: val})
    ytest.update({i: test})
    inputs.update({i: X})
    outputs.update({i: Y})
    lastvalues.update({i: np.array(last_values(train, timesteps=testsize))})
LocalM = LocalModel()
LocalM.fit(inputs, outputs)

train, val, test = split_train_val_test(df, testsize)
y = np.array(last_values(train, timesteps=testsize))

# global part
preds = []
for i in df.cluster.unique():
    preds.append(LocalM.predict(y, cluster=i))
totalpred = []
for i in range(0, clusterids.shape[0]):
    ypred = []
    for j in range(0, clusterids.shape[1]):
        ypred.append(local_weights[i, j] * preds[clusterids[i, j]][i])
    totalpred.append(np.sum(ypred, axis=0))
