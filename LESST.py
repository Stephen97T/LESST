from preprocessing import *
from timeseries import *
from tsforecast import *
from models import *
from data_prep import *
from clustering import *
import numpy as np
from sklearn.ensemble import RandomForestRegressor

np.random.seed(1)

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
inputs, outputs = prepare_inputoutput(df, testsize)
LocalM = LocalModel()
LocalM.fit(inputs, outputs)

train, val, test = split_train_val_test(df, testsize)
Gmodel = GlobalModel(LocalM, testsize, clusterids, local_weights)
Gmodel.fit(train, val)
# predict

finalpred = Gmodel.predict(train)
