import pandas as pd


def prepare_m4data(data, dataset_name, infopath="C:/thesis/data/m4"):
    seas_dict = {
        "Hourly": {
            "seasonality": 24,
            "input_size": 24,
            "output_size": 48,
            "freq": "H",
        },
        "Daily": {
            "seasonality": 7,
            "input_size": 7,
            "output_size": 14,
            "freq": "D",
        },
        "Weekly": {
            "seasonality": 52,
            "input_size": 52,
            "output_size": 13,
            "freq": "W",
        },
        "Monthly": {
            "seasonality": 12,
            "input_size": 12,
            "output_size": 18,
            "freq": "M",
        },
        "Quarterly": {
            "seasonality": 4,
            "input_size": 4,
            "output_size": 8,
            "freq": "Q",
        },
        "Yearly": {
            "seasonality": 1,
            "input_size": 4,
            "output_size": 6,
            "freq": "D",
        },
    }
    freq = seas_dict[dataset_name]["freq"]
    m4_info = pd.read_csv(
        infopath + "/M4-info.csv", usecols=["M4id", "category"]
    )
    m4_info = m4_info[
        m4_info["M4id"].str.startswith(dataset_name[0])
    ].reset_index(drop=True)
    df = data.copy()
    df = df.rename(columns={"V1": "unique_id"})
    df = pd.wide_to_long(
        df, stubnames=["V"], i="unique_id", j="ds"
    ).reset_index()
    df = df.rename(columns={"V": "y"})
    df = df.dropna()
    df["split"] = "train"
    df["ds"] = df["ds"] - 1
    # Get len of series per unique_id
    len_series = df.groupby("unique_id").agg({"ds": "max"}).reset_index()
    len_series.columns = ["unique_id", "len_serie"]
    len_series = df.groupby("unique_id").agg({"ds": "max"}).reset_index()
    dates = []
    for i in range(len(len_series)):
        len_serie = len_series.iloc[i, 1]
        ranges = pd.date_range(
            start="1970/01/01", periods=len_serie, freq=freq
        )
        dates += list(ranges)
    df.loc[:, "ds"] = dates

    df = df.merge(m4_info, left_on=["unique_id"], right_on=["M4id"])
    df.drop(columns=["M4id"], inplace=True)
    df = df.rename(columns={"category": "x"})

    X_train_df = df[df["split"] == "train"].filter(
        items=["unique_id", "ds", "x"]
    )
    y_train_df = df[df["split"] == "train"].filter(
        items=["unique_id", "ds", "y"]
    )

    X_train_df = X_train_df.reset_index(drop=True)
    y_train_df = y_train_df.reset_index(drop=True)
    return X_train_df, y_train_df
