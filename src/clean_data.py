import sys
import numpy as np
import pandas as pd
from pathlib import Path

BASE_PATH = Path(__file__)
src_dir = BASE_PATH.parent.parent.resolve()
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
data_path = src_dir.__str__() + "/data"
if str(data_path) not in sys.path:
    sys.path.insert(0, str(data_path))


def convert_to_p(nanomolar_value):
    return -np.log10(float(nanomolar_value) * 1e-9)


def clean_data(df):
    assert isinstance(df, pd.DataFrame)
    df_clean = df.dropna(subset=["Activity_Value", "SMILES", "Activity_Type"])
    valid_types = {"Ki", "Kd", "IC50"}
    df_filtered = df_clean[df_clean["Activity_Type"].isin(valid_types)]
    df_filtered["pActivity"] = df_filtered["Activity_Value"].apply(convert_to_p)
    df_filtered = df_filtered.dropna(subset=["pActivity"])
    df_ki = df_filtered[df_filtered["Activity_Type"] == "Ki"].copy()
    df_kd = df_filtered[df_filtered["Activity_Type"] == "Kd"].copy()
    df_ic50 = df_filtered[df_filtered["Activity_Type"] == "IC50"].copy()
    df_ki.reset_index(drop=True, inplace=True)
    df_kd.reset_index(drop=True, inplace=True)
    df_ic50.reset_index(drop=True, inplace=True)
    df_ki.to_csv(data_path + "/qsar_ki.csv")
    df_kd.to_csv(data_path + "/qsar_kd.csv")
    df_ic50.to_csv(data_path + "/qsar_ic50.csv")
    print("Done!")
    print(f"Ki entries: {df_ki.head()}")
    print(f"Kd entries: {df_kd.head()}")
    print(f"IC50 entries: {df_ic50.head()}")
    return df_ki, df_kd, df_ic50


df = pd.read_csv(data_path + "/qsar_dataset_async.csv", low_memory=False)
clean_data(df)
