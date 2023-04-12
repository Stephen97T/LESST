import pickle
import pandas as pd

with (
    open("C:/thesis/results/benchmark_Hourly_ds_True.pkl", "rb")
) as openfile:
    benchmark = pickle.load(openfile)

owa = {}
smape = {}
mase = {}
rmse = {}
for i in list(benchmark.keys()):
    if "OWA" in i:
        owa.update({i.split("_")[0][3:]: benchmark[i]})
    if "SMAPE" in i:
        smape.update({i.split("_")[0][3:]: benchmark[i]})
    if "MASE" in i:
        mase.update({i.split("_")[0][3:]: benchmark[i]})
    if "RMSE" in i:
        rmse.update({i.split("_")[0][3:]: benchmark[i]})

benchmark = pd.DataFrame(
    [owa, smape, mase, rmse], index=["OWA", "SMAPE", "MASE", "RMSE"]
)
benchmark.to_excel("C:/thesis/results/benchmark.xlsx")


c = {"3 "}
with (open("C:/thesis/results/lesst_Hourly_ds_False.pkl", "rb")) as openfile:
    season = pickle.load(openfile)

for i in list(benchmark.keys()):
    if "OWA" in i:
        res = benchmark[i]
        for j in list(res.keys()):
            naming = res[j.split("_")]

with (open("C:/thesis/results/lesst_Hourly_ds_True.pkl", "rb")) as openfile:
    deseason = pickle.load(openfile)
