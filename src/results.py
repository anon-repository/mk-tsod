import pandas as pd
import os
import lib
import numpy as np

def _combine(folder):
    files = sorted(os.listdir(folder))

    results = pd.DataFrame()
    for myfile in files:
        if "dtw" in myfile:
            continue
        df = pd.read_csv(folder + "/" + myfile, index_col=0)
        df = df.groupby("data").median()
        df["Experiment"] = myfile
        results = results.append(df)

    return results


def details(folder):
    return _combine(folder).pivot(columns="Experiment")


def ours(folder):
    return details(folder).rename({"my_auc": "roc_auc"}, axis=1)


if __name__ == "__main__":
    pass
