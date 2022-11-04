from models import LocalModel

loc = LocalModel("tsforecast", 12)
from preprocessing import *

df = read_m4_df("monthly")
feats = read_tsfeatures("monthly")
from clustering import *

clusters = FeatureClustering(n_clusters=10)
timeseries = clusters.cluster_features(feats, df)
distances = clusters.cluster_distances
iddistance = distances = clusters.idcluster_distance
iddistance = clusters.idcluster_distance
distances = clusters.cluster_distances
cluster1 = timeseries[timeseries.cluster == 1]
predictions = []
validation = []
for i in range(0, 100):
    oneserie = cluster1[cluster1.unique_id == cluster1.unique_id[i]]
    y = np.array(oneserie.y)[:-10]
    yval = np.array(oneserie.y)[-10:]
    validation.append(yval)
    timesteps = len(yval)
    loc.fit(y)
    predict = loc.ts_predict(timesteps)
    predictions.append(predict)
