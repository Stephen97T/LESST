from sklearn.cluster import KMeans
from timeseries import tsfeatures_r
from sklearn import preprocessing
import pandas as pd


class FeatureClustering:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.kmeans = ""
        self.clusters = ""
        self.midpoints = ""
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

    def cluster_features(self, tsfeatures, timeseries):
        tsfeatures, unique_ids = self.get_tsfeatures(tsfeatures)
        self.kmeans = self.cluster_kmeans(
            tsfeatures, n_clusters=self.n_clusters
        )
        self.clusters = self.kmeans.labels_
        self.midpoints = self.kmeans.cluster_centers_
        idmapping = {}
        for index in range(0, len(unique_ids)):
            idmapping.update({f"{unique_ids[index+1]}": self.clusters[index]})
        timeseries["cluster"] = timeseries.unique_id.map(idmapping)
        self.idmapping = idmapping
        return timeseries

    def get_tsfeatures(self, tsfeatures):
        tsfeatures, ids = (
            tsfeatures.drop("unique_id", axis=1),
            tsfeatures["unique_id"],
        )
        tsfeatures = preprocessing.normalize(tsfeatures, axis=0)
        return tsfeatures, ids


class DTWClustering:
    def __init__(self):
        pass
