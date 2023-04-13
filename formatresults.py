import pickle
import pandas as pd

path = "E:/documents/work/thesis"

with (open(f"{path}/benchmark_Hourly_ds_False.pkl", "rb")) as openfile:
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
benchmark.to_excel(f"{path}/benchmark.xlsx")


c = {"3 "}
with (open(f"{path}/lesst_Hourly_ds_False.pkl", "rb")) as openfile:
    season = pickle.load(openfile)


df = {}
for dataset in season:
    if "OWA" in dataset:
        dset = dataset.replace("ds:", "")
        tabel = {
            "3 clusters": {},
            "10 clusters": {},
            "30 clusters": {},
            "50 clusters": {},
            "100 clusters": {},
        }
        for models in season[dataset]:
            cname = models.split("_")
            rname = f"{cname[3].capitalize()}-{cname[5].capitalize()}"
            value = season[dataset][models]
            key = f"{cname[1]} clusters"
            tabel[key].update({rname: value})
        tabel = pd.DataFrame(tabel)
        df.update({dset: tabel})
        tabel.to_excel(f"E:/documents/work/thesis/LESST_{dset}_Seasonal.xlsx")

with (open(f"{path}/lesst_Hourly_ds_True.pkl", "rb")) as openfile:
    deseason = pickle.load(openfile)

dfs = {}
for dataset in deseason:
    if "OWA" in dataset:
        dset = dataset.replace("ds:", "")
        tabel = {
            "3 clusters": {},
            "10 clusters": {},
            "30 clusters": {},
            "50 clusters": {},
            "100 clusters": {},
        }
        for models in deseason[dataset]:
            cname = models.split("_")
            rname = f"{cname[3].capitalize()}-{cname[5].capitalize()}"
            value = deseason[dataset][models]
            key = f"{cname[1]} clusters"
            tabel[key].update({rname: value})
        tabel = pd.DataFrame(tabel)
        dfs.update({dset: tabel})
        tabel.to_excel(f"E:/documents/work/thesis/LESST_{dset}.xlsx")
