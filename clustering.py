from sklearn.cluster import KMeans
from timeseries import tsfeatures_r


class FeatureClustering:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.indeces = ""
        self.midpoints = ""

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

    def cluster_features(self, timeseries):
        tsfeatures = self.get_tsfeatures(timeseries)
        cluster = self.cluster_kmeans(tsfeatures, n_clusters=self.n_clusters)
        self.indeces = cluster.labels_
        self.midpoints = cluster.cluster_centers_
        clusterseries = []
        for cluster in range(0, self.n_cluster):
            clusterseries.append(timeseries[self.indeces == cluster])
        return cluster
    
    def get_tsfeatures(self,timeseries):
        


class DTWClustering:
    def __init__(self):
        pass
