from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import (
    LinearRegression,
    HuberRegressor,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from models import WeightedSum
from LESST import LESST
from benchmark import BenchmarkModel, PerformanceMeasures
from tsforecast import ThetaF
import numpy as np

# models to consider
xgb = XGBRegressor(tree_method="gpu_hist")
lgbm = LGBMRegressor()
rf = RandomForestRegressor(
    bootstrap=True, max_depth=80, max_features=6, n_estimators=100
)
ols = LinearRegression()
huber = HuberRegressor()
gradient = GradientBoostingRegressor()


dataset = "yearly"
n_clusters = 60
horizon = 3
frequency = 1
localmodel = ols
globalmodel = huber

less = LESST(dataset, n_clusters)
less.fit(
    prediction_steps=horizon, localmodel=localmodel, globalmodel=globalmodel
)

predictions = less.predict()
leni = len(predictions) / horizon
predictions = predictions.reshape(int(leni), horizon)

train = less.train
val = less.val
test = less.test

total_train = []
for i in range(0, len(train)):
    total_train.append(np.concatenate([train[i], val[i]]))

benchmark_owa = []
lesst_owa = []

for i in range(0, len(total_train)):
    benchmark = BenchmarkModel(ThetaF(frequency))
    bench_owa = benchmark.performance(test[i], train[i], horizon, frequency)
    measure = PerformanceMeasures(frequency)
    mod_owa = measure.OWA(test[i], predictions[i], train[i])
    benchmark_owa.append(bench_owa)
    lesst_owa.append(mod_owa)

bench = np.array(benchmark_owa)
lesst = np.array(lesst_owa)
bench[bench >= 1e308] = 0
lesst[lesst >= 1e308] = 0
print(bench.mean())
print(lesst.mean())
