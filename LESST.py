from preprocessing import read_tsfeatures, read_m4_df
from models import LocalModel, GlobalModel
from data_prep import split_train_val_test, prepare_inputoutput
from clustering import FeatureClustering
import numpy as np
from xgboost import XGBRegressor
from time import time

np.random.seed(1)


class LESST:
    def __init__(self, dataset, n_clusters):
        self.n_clusters = n_clusters
        self.df = read_m4_df(dataset)
        self.feats = read_tsfeatures(dataset).reset_index(drop=True)
        self.df = self.df[
            self.df.unique_id.isin(self.df.unique_id.unique()[0:1000])
        ].reset_index(drop=True)
        self.feats = self.feats.loc[0:999].reset_index(drop=True)
        """
        self.df = self.df[
            self.df.unique_id.isin(["M1", "M2", "M3", "M4"])
        ].reset_index(drop=True)
        self.feats = self.feats[
            self.feats.unique_id.isin(["M1", "M2", "M3", "M4"])
        ].reset_index(drop=True)
        """

    def fit(
        self,
        prediction_steps,
        localmodel=XGBRegressor(tree_method="gpu_hist"),
        globalmodel=XGBRegressor(tree_method="gpu_hist"),
    ):
        t = time()
        clust = FeatureClustering(self.n_clusters)
        clust.cluster_features(self.feats, self.df)
        local_weights = clust.cluster_distances
        clusterids = clust.idcluster_distance
        print(f"clustering step took {time()-t} sec")
        tt = time()
        testsize = prediction_steps
        inputs, outputs = prepare_inputoutput(self.df, testsize)
        print(f"local model dataprep took {time()-tt} sec")
        tt = time()
        self.LocalM = LocalModel(modeltype=localmodel)
        self.LocalM.fit(inputs, outputs)
        print(f"local model step took {time()-tt} sec")

        tt = time()
        self.train, self.val, self.test = split_train_val_test(
            self.df, testsize
        )
        print(f"datasplit part took {time()-tt} sec")
        tt = time()
        self.Gmodel = GlobalModel(
            self.LocalM,
            testsize,
            clusterids,
            local_weights,
            model=globalmodel,
        )
        self.Gmodel.fit(self.train, self.val)
        print(f"global model part took {time()-tt} sec")
        print(f"total running time {time()-t} sec")

    def predict(self):
        predictions = self.Gmodel.predict(self.val)
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
