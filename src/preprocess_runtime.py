import os
import numpy as np
import pandas as pd
import sklearn.utils

df = pd.read_csv("../data/preprocessed/0.05/Lightning2/0.csv", index_col=0)

len_series = df.shape[1]

for target_length in range(50, len_series, 50):

    out_path = f"../data/preprocessed/runtime_various_length_of_ts/{target_length}/"

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for i in range(5):

        # -1 to omit the "class" column
        idx = sorted(np.random.choice(range(0, len_series-1),
                                      size=target_length, replace=False))

        sampled = df.iloc[:, idx + [len_series-1]]

        sampled.to_csv(
            f"../data/preprocessed/runtime_various_length_of_ts/{target_length}/{i}.csv")

for n_samples in range(100, 1000+1, 100):
    out_path = f"../data/preprocessed/runtime_various_number_of_samples/{n_samples}/"

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for i in range(5):

        sampled = sklearn.utils.resample(df, n_samples=n_samples)
        X = sampled.drop("class", axis=1)
        y = sampled["class"]

        X += np.random.normal(scale=10e-3, size=X.shape)
        X["class"] = y
        X = X.reset_index(drop=True)
        X.to_csv(
            f"../data/preprocessed/runtime_various_number_of_samples/{n_samples}/{i}.csv")
