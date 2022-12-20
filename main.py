from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from models import WeightedSum
from LESST import LESST
from benchmark import BenchmarkModel, PerformanceMeasures
from tsforecast import ThetaF
import numpy as np

dataset = "monthly"
n_clusters = 4
horizon = 12
frequency = 12
localmodel = LinearRegression()
globalmodel = LinearRegression()

less = LESST(dataset, n_clusters)
less.fit(
    prediction_steps=horizon, localmodel=localmodel, globalmodel=globalmodel
)

predictions = less.predict()
predictions = less.reshape(1000, 12)

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
