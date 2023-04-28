"""This file contains the LESST model"""
from preprocessing import read_tsfeatures
from models import LocalModel, GlobalModel
from data_prep import split_train_val, prepare_inputoutput
from clustering import FeatureClustering
import numpy as np
from xgboost import XGBRegressor
from time import time

np.random.seed(1)


class LESST:
    def __init__(
        self,
        dataset,
        df,
        n_clusters,
        freq,
        deseason,
        split=True,
        start=True,
        rolling=False,
        evenweight=False,
    ):
        """Initiate parameters for the LESST model"""
        self.n_clusters = n_clusters
        self.feats = read_tsfeatures(dataset).reset_index(drop=True)
        self.df = df
        self.freq = freq
        self.deseason = deseason
        self.split = split
        self.start = start
        self.rolling = rolling
        self.evenweight = evenweight

    def fit(
        self,
        prediction_steps,
        localmodel=XGBRegressor(tree_method="gpu_hist"),
        globalmodel=XGBRegressor(tree_method="gpu_hist"),
    ):
        """Fit the LESST model"""

        # Set prediction steps
        self.steps = prediction_steps
        t = time()

        # Start Timeseries Feature Clustering
        clust = FeatureClustering(self.n_clusters)
        clust.cluster_features(self.feats, self.df, self.steps)

        # Set local weights
        local_weights = clust.cluster_distances
        clusterids = clust.idcluster_distance
        self.cluster_idmap = clust.idmapping
        print(f"clustering step took {time()-t} sec")
        tt = time()
        testsize = prediction_steps

        # Create input and target set for fitting Local Models
        inputs, outputs = prepare_inputoutput(
            self.df, testsize, self.freq, self.deseason, self.split, self.start
        )
        print(f"local model dataprep took {time()-tt} sec")
        tt = time()

        # Initiate and fit Local Models
        self.LocalM = LocalModel(modeltype=localmodel)
        self.LocalM.fit(inputs, outputs)
        print(f"local model step took {time()-tt} sec")

        tt = time()

        # Split data for the Global Model incase rolling
        # Then all data in the set will be used
        if self.rolling:
            split = False
        else:
            split = True
        self.train, self.val, self.seas = split_train_val(
            self.df.drop("cluster", axis=1),
            testsize,
            self.freq,
            self.deseason,
            split=split,
        )
        print(f"datasplit part took {time()-tt} sec")
        tt = time()

        # Initiate and fit Global Model
        self.Gmodel = GlobalModel(
            self.LocalM,
            testsize,
            clusterids,
            local_weights,
            model=globalmodel,
        )
        self.Gmodel.fit(
            self.train,
            self.val,
            self.rolling,
            self.evenweight,
        )
        print(f"global model part took {time()-tt} sec")
        print(f"total running time {time()-t} sec")

    def predict(self):
        """Make predictions using the LESST model"""
        predictions = self.Gmodel.predict(self.val, self.evenweight)

        # Reseasonalize predictions in case they were deseasonalized
        if self.deseason:
            preds = []
            for i in range(0, int(len(predictions) / self.steps)):
                pred = np.reshape(
                    predictions[i * self.steps : (i + 1) * self.steps],
                    self.steps,
                )
                preds.append(self.seas[i].reseasonalize_pred(pred))
        else:
            preds = predictions.reshape(
                int(len(predictions) / self.steps), self.steps
            )
        return np.array(preds)
