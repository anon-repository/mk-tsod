Time Series Outlier Detection with Support Vector Data Description (TSOD-SVDD)
===============================================================================================

This repository hosts the supplementary material for the paper:

- TSOD-SVDD: Time Series Outlier Detection with Support Vector Data Description. (Under review).

The repository contains our code, instructions to reproduce our experiments, and additional information about the algorithm. 

We use publicly available data from the [UCR Time Series Classification Archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/).

We release our code with an AGPLv3 license. If you are using code from this repository, please cite our paper.

Supplementary Material
------------------------

The supplementary material is under review and still unreleased.

Instructions for reproducing the experiments
----------------

Follow the steps below to reproduce the results of TSOD. We give the random seeds, to ensure reproducibility. We do not share the source code of the competitors, which may be obtained from the respective sources, or by contacting the authors. We include our result files, for ease of viewing with `visualize.ipynb`.

1. Clone this repository.
1. Obtain the data from UCR per instructions on their [webpage](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/).

    	cp UCRArchive_2018.zip tsod/data/raw/
		cd tsod/data/raw
		unzip UCRArchive_2018.zip (requires password)
	
1. Install dependencies.

    	conda create -n tsod python=3.7 pandas=1.3.3 scikit-learn=1.0 numexpr=2.7.3 jupyter numpy=1.21.5
		conda activate tsod
		pip install tsfresh dtaidistance
    	pip install git+https://github.com/anon-repository/tsvdd.git@main
	
1. Create train data / CV folds for outlier detection.

        cd tsod/src
	    python preprocess.py
	
1. Create data for runtime experiments.

        cd tsod/src
		python preprocess_runtime.py
	
1. Run the algorithm.

		cd tsod/src
		python run.py
		# to run runtime experiments, uncomment the code in `run.py`.
	
1. View the results.

		cd tsod
		jupyter notebook
		# open notebooks/visualize.ipynb

Acknowledgements
-----------------

- We base our SimpleMKL implementation on Wenyi Qin's, available [here](https://github.com/qintian0321/SimpleMKL_python).

