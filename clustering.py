"""This file contains the "timeseries features clustering method"""

from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from sklearn import preprocessing
import numpy as np


class FeatureClustering:
    def __init__(self, n_clusters):
        """Set initial number of clusters for FeatureClustering object"""
        self.n_clusters = n_clusters
        self.kmeans = ""
        self.clusters = ""
        self.cluster_distances = ""
        self.idcluster_distance = ""
        self.midpoints = ""
        self.tree = ""
        self.idmapping = {}

    def cluster_kmeans(self, timeseries, n_clusters):
        """Fit using the Kmeans method"""
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
        """Cluster the timeseries for a whole dataset using the
        Timseries Features"""

        # Get timeseries features
        tsfeatures, unique_ids = self.get_tsfeatures(tsfeatures)

        # Cluster the features for all series
        self.kmeans = self.cluster_kmeans(
            tsfeatures, n_clusters=self.n_clusters
        )

        # Save the allocation of clusters
        self.clusters = self.kmeans.labels_

        # Save the cluster centers
        self.midpoints = self.kmeans.cluster_centers_

        # Initialize KDTree method
        self.tree = KDTree(self.midpoints)

        # Calculate distances for each timeseries to the clusters
        cluster_distances = self.tree.query(tsfeatures, k=self.n_clusters)

        # Save the order of cluster distances
        self.idcluster_distance = cluster_distances[1]

        # Incase distance is 0 set very small
        cluster_distances[0][cluster_distances[0] == 0] = np.power(np.e, -100)

        # Calculate weights
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

        # Set cluster number for each series
        idmapping = {}
        for index in range(0, len(unique_ids)):
            idmapping.update({f"{unique_ids[index]}": self.clusters[index]})
        timeseries["cluster"] = timeseries.index.map(idmapping)
        self.idmapping = idmapping
        return timeseries

    def get_tsfeatures(self, tsfeatures):
        """Reformat the timeseries feature data"""
        tsfeatures, ids = (
            tsfeatures.drop("unique_id", axis=1),
            tsfeatures["unique_id"],
        )
        tsfeatures = tsfeatures.dropna(axis=1)
        tsfeatures = preprocessing.normalize(tsfeatures, axis=0)
        return tsfeatures, ids
