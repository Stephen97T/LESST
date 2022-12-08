from preprocessing import read_tsfeatures, read_m4_df
from timeseries import *
from tsforecast import *
from models import LocalModel, GlobalModel, WeightedSum
from data_prep import split_train_val_test, prepare_inputoutput
from clustering import FeatureClustering
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

np.random.seed(1)


class LESST:
    def __init__(self, dataset, n_clusters):
        self.n_clusters = n_clusters
        self.df = read_m4_df("monthly")
        self.feats = read_tsfeatures("monthly")
        self.df = self.df[
            self.df.unique_id.isin(["M1", "M2", "M3", "M4"])
        ].reset_index(drop=True)
        self.feats = self.feats[
            self.feats.unique_id.isin(["M1", "M2", "M3", "M4"])
        ].reset_index(drop=True)

    def fit(self, prediction_steps):
        clust = FeatureClustering(self.n_clusters)
        clust.cluster_features(self.feats, self.df)
        local_weights = clust.cluster_distances
        clusterids = clust.idcluster_distance

        testsize = prediction_steps
        inputs, outputs = prepare_inputoutput(self.df, testsize)
        LocalM = LocalModel()
        LocalM.fit(inputs, outputs)

        self.train, self.val, self.test = split_train_val_test(
            self.df, testsize
        )
        self.Gmodel1 = GlobalModel(
            LocalM,
            testsize,
            clusterids,
            local_weights,
            model=RandomForestRegressor(),
        )
        self.Gmodel1.fit(self.train, self.val)
        self.Gmodel2 = GlobalModel(
            LocalM, testsize, clusterids, local_weights, model=XGBRegressor()
        )
        self.Gmodel2.fit(self.train, self.val)
        self.Gmodel3 = GlobalModel(
            LocalM, testsize, clusterids, local_weights, model=WeightedSum()
        )
        self.Gmodel3.fit(self.train, self.val)

    def predict(self):
        predictions = self.Gmodel3.predict(self.val)
        return predictions


# predict

# finalpred1mod1 = Gmodel1.predict(train)
# finalpred2mod1 = Gmodel1.predict(val)
# finalpred1mod2 = Gmodel2.predict(train)
# finalpred2mod2 = Gmodel2.predict(val)
# finalpred1mod3 = Gmodel3.predict(train)
# finalpred2mod3 = Gmodel3.predict(val)
# val = np.array(val).reshape(-1, 1)
# test = np.array(test).reshape(-1, 1)
