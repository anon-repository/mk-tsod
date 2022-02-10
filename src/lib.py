import numpy as np
import numexpr as ne

from sklearn.metrics import cohen_kappa_score, roc_curve, auc, matthews_corrcoef, balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score, precision_recall_curve
from tsfresh.utilities.dataframe_functions import impute

from tsvdd.utils import sampled_gak_sigma
from tsvdd.kernels import train_kernel_matrix
from tsfresh import extract_features
import sys


def rbf_kernel_fast_ghafoori(X, gamma):
    """
    Computes the gram RBF Kernel matrix

    Inspired by https://stackoverflow.com/questions/47271662/what-is-the-fastest-way-to-compute-an-rbf-kernel-in-python.
    """
    X_norm = -np.einsum('ij,ij->i', X, X)
    return ne.evaluate('exp(g * (A + B + 2 * C))', {
        'A': X_norm[:, None],
        'B': X_norm[None, :],
        'C': np.dot(X, X.T),
        'g': gamma
    })


def rbf_kernel_fast(X, sigma):
    """
    Computes the gram RBF Kernel matrix

    Inspired by https://stackoverflow.com/questions/47271662/what-is-the-fastest-way-to-compute-an-rbf-kernel-in-python.
    """
    X_norm = -np.einsum('ij,ij->i', X, X)
    gamma = 1 / (2.0 * sigma ** 2)
    return ne.evaluate('exp(g * (A + B + 2 * C))', {
        'A': X_norm[:, None],
        'B': X_norm[None, :],
        'C': np.dot(X, X.T),
        'g': gamma
    })


def get_gamma(X):
    """https://people.eng.unimelb.edu.au/smonazam/publications/Ghafoori2018TNNLS.pdf"""
    n = len(X)
    distances = np.zeros((n, n))

    delta_min = sys.maxsize
    delta_max = 0
    q = 0

    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(X.iloc[i] - X.iloc[j])
            distances[i, j] = dist
            distances[j, i] = dist

            if distances[i, j] < delta_min:
                delta_min = distances[i, j]
                q = i

    delta_max = np.sum(distances[q])/(n-1)

    return -np.log(delta_min/delta_max)/(delta_max**2-delta_min**2)


def my_extract_features(X, features):
    X_train = X.copy()
    X_train["id"] = X_train.index
    X_train = X_train.melt(id_vars="id", var_name="time")
    X_train["time"] = X_train["time"].astype(int)

    X_train = X_train.sort_values(
        ["id", "time"]).reset_index(drop=True)

    X_train_features = extract_features(
        X_train, default_fc_parameters=features, column_id="id", column_sort="time", impute_function=impute, disable_progressbar=True, n_jobs=0)

    return X_train_features.apply(normalize_0_1, axis=0).fillna(
        0)  # for constant values, normalize_0_1 returns nan;


def rbf_kernel_ghafoori(feature_matrix):
    n_instances = len(feature_matrix)
    K = np.ones((n_instances, n_instances))
    gamma = get_gamma(feature_matrix.drop_duplicates())
    for feature in feature_matrix.values.T:
        K *= rbf_kernel_fast_ghafoori(feature.reshape((-1, 1)), gamma)
    return K


def get_combined_kernel(kernel_matrices, weights):
    M = len(kernel_matrices)
    n = kernel_matrices[0].shape[0]

    combined_kernel_matrix = np.zeros((n, n))
    for m in range(M):
        combined_kernel_matrix += kernel_matrices[m] * weights[m] * 1.0

    return combined_kernel_matrix


def normalize(df):
    return ((df.T-df.T.mean())/df.T.std()).T


def optimal_C(y):
    return 1/np.sum(y == -1)


def normalize_0_1(data):
    """
    Returns normalized data between 0 and 1.

    :param data: numpy or DataFrame
    :return:
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def calc_metrics(y_true, y_pred, y_score):
    metrics = {'MCC': 0.0, 'Kappa': 0.0, 'Accuracy': 0.0, 'Sensitivity': 0.0,
               'Specificity': 0.0, 'AUC': 0.0, 'BA': 0.0}

    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    metrics['Kappa'] = cohen_kappa_score(y_true, y_pred)
    metrics['Sensitivity'] = sensitivity_outlier(y_true, y_pred)
    metrics['Specificity'] = specificity_outlier(y_true, y_pred)
    metrics['BA'] = balanced_accuracy_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=-1)
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['AUC'] = auc(fpr, tpr)
    metrics['f1_score'] = f1_score(y_true, y_pred)

    if y_score is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        metrics["pr_auc"] = auc(recall, precision)
    return metrics


def specificity_outlier(y_true, y_pred):
    normal = y_true == 1
    true_normal = (y_pred == 1) & (normal)

    spec = np.sum(true_normal) / np.sum(normal)
    return spec


def sensitivity_outlier(y_true, y_pred):
    outlier = y_true == -1
    true_outlier = (y_pred == -1) & (outlier)

    sen = np.sum(true_outlier) / np.sum(outlier)
    return sen


def ga_gram(X_train, multiplier, triang=None):
    n_instances = X_train.shape[0]
    X_train_tgak = np.reshape(
        X_train.values, (X_train.shape[0], X_train.shape[1], 1), order='C')
    X_train_tgak = np.ascontiguousarray(X_train_tgak, dtype=np.float64)

    tga_sigma = sampled_gak_sigma(X_train.values, n_samples=min(
        n_instances, 200), multipliers=[multiplier])[0]
    if triang is None:
        triangular = int(X_train.shape[1]/2)
    else:
        triangular = triang
    K_GA_train = train_kernel_matrix(
        X_train_tgak, tga_sigma, triangular, 'exp')
    return K_GA_train
