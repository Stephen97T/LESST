import pandas as pd
import pickle

with open("D:/documents/thesisresults/benchmark_Quarterly_ds_True.pkl", "rb") as f:
    benchmarks = pickle.load(f)

with open("D:/documents/thesisresults/benchmark_Hourly_ds_True.pkl", "rb") as f:
    benchmarks.update(pickle.load(f))
benchmark = pd.DataFrame([benchmarks])
benchmark.columns = benchmark.columns.str.replace("ds:", "")
benchmark.to_excel("D:/documents/thesisresults/benchmark.xlsx")

with open("D:/documents/thesisresults/lesst_Quarterly_ds_True.pkl", "rb") as f:
    lesst = pickle.load(f)

with open("D:/documents/thesisresults/lesst_Hourly_ds_False.pkl", "rb") as f:
    lesst.update(pickle.load(f))

df = {}
for dataset in lesst:
    dset = dataset.replace("ds:", "")
    tabel = {
        "3 clusters": {},
        "10 clusters": {},
        "30 clusters": {},
        "50 clusters": {},
        "100 clusters": {},
    }
    for models in lesst[dataset]:
        cname = models.split("_")
        rname = f"{cname[3].capitalize()}-{cname[5].capitalize()}"
        value = lesst[dataset][models]
        key = f"{cname[1]} clusters"
        tabel[key].update({rname: value})
    tabel = pd.DataFrame(tabel)
    df.update({dset: tabel})
    tabel.to_excel(f"D:/documents/thesisresults/LESST_{dset}_Seasonal.xlsx")
