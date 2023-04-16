from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans
from scipy.spatial import KDTree
from timeseries import tsfeatures_r
from sklearn import preprocessing
import pandas as pd
import numpy as np


class FeatureClustering:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.kmeans = ""
        self.clusters = ""
        self.cluster_distances = ""
        self.idcluster_distance = ""
        self.midpoints = ""
        self.tree = ""
        self.idmapping = {}

    def cluster_kmeans(self, timeseries, n_clusters):
        kmeans = KMeans(
            init="random",
            n_clusters=n_clusters,
            n_init=10,
            max_iter=300,
            random_state=42,
        )
        kmeans.fit(timeseries)
        return kmeans

    def cluster_features(self, tsfeatures, timeseries, timesteps):
        tsfeatures, unique_ids = self.get_tsfeatures(tsfeatures)
        self.kmeans = self.cluster_kmeans(
            tsfeatures, n_clusters=self.n_clusters
        )
        self.clusters = self.kmeans.labels_
        self.midpoints = self.kmeans.cluster_centers_
        self.tree = KDTree(self.midpoints)
        cluster_distances = self.tree.query(tsfeatures, k=self.n_clusters)
        self.idcluster_distance = cluster_distances[1]
        cluster_distances[0][cluster_distances[0] == 0] = np.power(np.e, -100)
        self.cluster_distances = (
            cluster_distances[0].sum(axis=1) / cluster_distances[0].T
        )
        self.cluster_distances = (
            self.cluster_distances / self.cluster_distances.sum(axis=0)
        ).T
        weights_ordered = []
        for weight, order in zip(
            self.cluster_distances, self.idcluster_distance
        ):
            weights_ordered.append(weight[np.argsort(order)])
        self.cluster_distances = np.repeat(
            np.array(weights_ordered), timesteps, axis=0
        )
        idmapping = {}
        for index in range(0, len(unique_ids)):
            idmapping.update({f"{unique_ids[index]}": self.clusters[index]})
        timeseries["cluster"] = timeseries.index.map(idmapping)
        self.idmapping = idmapping
        return timeseries

    def get_tsfeatures(self, tsfeatures):
        tsfeatures, ids = (
            tsfeatures.drop("unique_id", axis=1),
            tsfeatures["unique_id"],
        )
        tsfeatures = tsfeatures.dropna(axis=1)
        tsfeatures = preprocessing.normalize(tsfeatures, axis=0)
        return tsfeatures, ids
