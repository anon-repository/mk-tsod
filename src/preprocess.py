import os
import json
import pandas as pd
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

project_dir = ".."
outlier_ratio = 0.05
NORMAL_DATA_RATIO = 1

CONFIG_FILE = './datasets.json'
DATASETS = ['ECG200', 'GunPoint', 'ECGFiveDays', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'CBF', 'Coffee', 'FaceFour', 'Ham', 'Herring', 'Lightning2', 'Lightning7', 'Meat', 'MedicalImages', 'MoteStrain',
            'Plane', 'Strawberry', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'Wine', 'ChlorineConcentration', 'Symbols', 'Wafer']
RANDOM_SEEDS = [1803, 4949, 8727, 1773, 7276, 1847, 9780, 7850, 7457, 8074]


def get_files(there: str, file_extension: str):
    """
    Returns csv files in directory 'there'.

    :param there: relative path
    :param file_extension:
    :return: list with csv files
    """
    path = project_dir + '/' + there
    files = [path + x for x in os.listdir(path) if file_extension in x]
    return sorted(files)


def _get_dataset_config():
    with open(CONFIG_FILE) as f:
        data = json.load(f)
    return data


def load_dataset(name: str):
    """
    Load dataset with config from CONFIG_FILE. Returns files alphabetically.
    :param name: dataset name
    :return: list of data frames
    """
    data_file = _get_dataset_config()
    dataset_dict = data_file[name]
    file_extension = dataset_dict['file_extension']
    files = get_files(dataset_dict['path'], file_extension)

    if not files:
        breakpoint()
        raise FileNotFoundError
    dfs = []
    for file in files:
        dfs.append(pd.read_csv(
            file, **dataset_dict['csv_options'], dtype='float'))
    return dfs


def beggel_cv(normal_index, outlier_index, outlier_ratio, normal_data_ratio, random_states):
    """
    Given the random_states, sampled train test splits are created.
    Should be reproducible, seed is set in sample_data()

    :param normal_index:
    :param outlier_index:
    :param random_states:
    :param outlier_ratio:
    :yield: [(np.array, np.array)], [(np.array, np.array)]
    """

    n_normal_train = int(len(normal_index) * normal_data_ratio)
    n_train = int(n_normal_train / (1 - outlier_ratio))
    n_outlier_train = n_train - n_normal_train
    #n_outlier_train = n_outlier_train if n_outlier_train > 0 else 1

    for random_state in random_states:
        train_normal_index = random_state.choice(
            normal_index.values, n_normal_train, replace=False)
        train_normal_index = pd.Int64Index(train_normal_index)
        train_outlier_index = random_state.choice(
            outlier_index.values, n_outlier_train, replace=False)
        train_outlier_index = pd.Int64Index(train_outlier_index)

        train = train_normal_index.union(train_outlier_index)
        if normal_data_ratio == 1:
            yield train
            continue
        # get normal data
        # drop normal-X_train from normal data
        test_normal_index = normal_index.drop(train_normal_index)
        # remaining data is normal and not in X_train

        # get outlier data
        # drop normal-X_train from normal data
        test_outlier_index = outlier_index.drop(train_outlier_index)
        # remaining data is abnormal and not in X_train

        test = test_normal_index.union(test_outlier_index)

        yield train, test


def load_dataframe(name: str):
    """
    Load dataset with config from CONFIG_FILE.
    Returns dataframe with labels -1 for outlier and 1 for normals.
    :param name: dataset name
    :return: DataFrame
    """
    dfs = load_dataset(name)
    data_file = _get_dataset_config()
    dataset_dict = data_file[name]
    normal_labels = dataset_dict['normal_labels']
    if len(normal_labels) == 0:
        raise ValueError(
            'dataset.json is invalid. Must contain at least one normal class label.')
    outlier_labels = dataset_dict['outlier_labels']
    if len(outlier_labels) == 0:
        raise ValueError(
            'dataset.json is invalid. Must contain at least one outlier class label.')
    if 1000 in normal_labels or -1000 in outlier_labels:
        raise ValueError("You can't be serious!")
    for df in dfs:
        for normal_label in normal_labels:
            normal_index = df[df.iloc[:, 0] == float(normal_label)].index
            df.iloc[normal_index, 0] = 1000
        for outlier_label in outlier_labels:
            outlier_index = df[df.iloc[:, 0] == float(outlier_label)].index
            df.iloc[outlier_index, 0] = -1000
        normal_index = df[df.iloc[:, 0] == float(1000)].index
        df.iloc[normal_index, 0] = 1
        outlier_index = df[df.iloc[:, 0] == float(-1000)].index
        df.iloc[outlier_index, 0] = -1
    return pd.concat(dfs, ignore_index=True)


def get_random_states():
    random_states = []
    for seed in RANDOM_SEEDS:
        random_states.append(RandomState(MT19937(SeedSequence(seed))))
    return random_states


for data_set_i, data_set in enumerate(DATASETS):
    print(f"{data_set_i}: Dataset: {data_set}", flush=True)
    df = load_dataframe(data_set)
    normal_index = df[df.iloc[:, 0] == 1].index
    outlier_index = df[df.iloc[:, 0] == -1].index
    random_states = get_random_states()
    cv_i = 1

    results_list = []
    for train_index in beggel_cv(normal_index, outlier_index, outlier_ratio, NORMAL_DATA_RATIO, random_states):
        test_index = []
        print(f'CV-Split: {cv_i}', flush=True)
        iter_result = dict()
        iter_result['cv_split'] = cv_i
        cv_i += 1
        iter_result['data_set'] = data_set
        n_instances = len(train_index)
        iter_result['n_instances'] = n_instances
        iter_result['outlier_ratio'] = outlier_ratio
        iter_result['normal_data_ratio'] = NORMAL_DATA_RATIO
        X_train = df.iloc[train_index, 1:]
        X_test = df.iloc[test_index, 1:]
        y_train = df.iloc[train_index, 0]
        y_test = df.iloc[test_index, 0]
        if not os.path.exists(f"../data/preprocessed/{outlier_ratio}/{data_set}"):
            os.mkdir(f"../data/preprocessed/{outlier_ratio}/{data_set}")
        X_train["class"] = y_train
        X_train.to_csv(
            f"../data/preprocessed/{outlier_ratio}/{data_set}/{cv_i-2}.csv")
