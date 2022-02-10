from lib import rbf_kernel_fast, normalize_0_1, calc_metrics, ga_gram, my_extract_features, optimal_C, get_combined_kernel  # noqa
from tsvdd.SVDD import SVDD

import time

import pandas as pd
import numpy as np
import os

import experiments


def cv(experiment, datasets, dataset_path, output_path_prefix, runs=10):
    result_filename = f"{output_path_prefix}/{experiment.get_name()}.csv"
    #result_filename = f"../../results/runtime_various_number_of_samples/tsod-svdd/{experiment.get_name()}.csv"
    print("Running: ", experiment.get_name())
    result = pd.DataFrame()

    for dataset in datasets:
        for cv in range(runs):
            print(experiment.get_name(), dataset, cv)
            df = pd.read_csv(
                f"{dataset_path}/{dataset}/{cv}.csv", index_col=0)
            X = df.drop("class", axis=1)
            y = df["class"].values

            clf = SVDD(kernel="precomputed",
                       C=experiment.optimal_C(y), tol=1e-10, verbose=False)
            train_start = time.time()
            K = experiment.kernel(X, y)

            clf.fit(K)
            train_end = time.time()
            y_pred = clf.predict(K, np.ones(len(K))).reshape(-1)
            prediction_end = time.time()

            y_score = clf.predict(K, np.ones(
                len(K)), dec_vals=True).reshape(-1)
            metrics = experiment.metrics(y, y_pred, y_score*-1)
            metrics["cv"] = cv
            metrics["data"] = dataset
            metrics["train_time"] = train_end-train_start
            metrics["prediction_time"] = prediction_end - train_end

            result = result.append(metrics, ignore_index=True)

        result.to_csv(result_filename)


def main():
    experiment = experiments.MKL(
        experiments.GAK(), experiments.FFTRBFGhafoori(10), 0.06)

    ##################################
    ## Outlier Detection Experiment ##
    ##################################
    input_files = ["ArrowHead", "CBF", "ChlorineConcentration", "ECG200", "ECGFiveDays", "GunPoint", "Ham", "Herring",
                   "Lightning2", "MoteStrain", "Strawberry", "Symbols", "ToeSegmentation1", "ToeSegmentation2", "TwoLeadECG", "Wafer", "Wine"]
    cv(experiment, input_files, dataset_path="../data/preprocessed/0.05",
       output_path_prefix="../results/0.05/", runs=10)

    ########################
    ## Runtime Experiment ##
    ########################
#    runtime_length_ts_files = [*range(50, 650, 50)]
#    cv(experiment, runtime_length_ts_files, dataset_path="../data/preprocessed/runtime_various_length_of_ts",
#       output_path_prefix="../results/runtime_various_length_of_ts/", runs=1)
#
#    runtime_num_samples_files = [*range(100, 1000+1, 100)]
#    cv(experiment, runtime_num_samples_files, dataset_path="../data/preprocessed/runtime_various_number_of_samples",
#       output_path_prefix="../results/runtime_various_number_of_samples/", runs=1)
#


if __name__ == "__main__":
    main()
