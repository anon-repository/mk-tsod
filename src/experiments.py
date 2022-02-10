from sklearn.metrics import roc_auc_score, balanced_accuracy_score, auc, precision_recall_curve  # noqa
from simple_mkl_svdd import mkl_svdd
import numpy as np

from lib import rbf_kernel_fast, normalize_0_1, calc_metrics, ga_gram, my_extract_features, optimal_C, get_combined_kernel, rbf_kernel_ghafoori  # noqa


def fix_d(d, cutoff=0.1):
    if d[0] < cutoff:
        return [cutoff, 1-cutoff]
    if d[1] < cutoff:
        return [1-cutoff, cutoff]
    return d


class Experiment:
    def __init__(self):
        pass

    def kernel(self, data, y):
        pass

    def optimal_C(self, y, outlier_ratio=None):
        return optimal_C(y)

    def metrics(self, y, y_pred, y_score):
        return calc_metrics(y, y_pred, y_score)

    def get_name(self):
        return "please-name-the-experiment"


class GAK(Experiment):
    def kernel(self, data, y):
        return ga_gram(data, multiplier=1.5)

    def get_name(self):
        return "gak-"


class FFTRBFGhafoori(Experiment):

    def __init__(self, fft_count, use_diff=False):
        self.fft_count = fft_count
        self.fc = {'fft_coefficient': [
            {'attr': 'real', 'coeff': i} for i in range(1, self.fft_count)]}
        self.use_diff = use_diff

    def kernel(self, data, y):
        fe = my_extract_features(data, self.fc)
        fe = fe.loc[:, (fe != 0).any(axis=0)]
        if self.use_diff:
            fe = fe.diff(axis=1).fillna(0)
            fe = fe.loc[:, (fe != 0).any(axis=0)]

        return rbf_kernel_ghafoori(fe)

    def get_name(self):
        if self.use_diff:
            return "fftrbf-ghafoori-" + str(self.fft_count) + "-diff-"
        return "fftrbf-ghafoori-" + str(self.fft_count) + "-"


class MKL(Experiment):
    def __init__(self, ex1, ex2, cutoff=0):
        self.ex1 = ex1
        self.ex2 = ex2
        self.cutoff = cutoff

    def kernel(self, data, y):
        k1 = self.ex1.kernel(data, y)
        k2 = self.ex2.kernel(data, y)

        C = self.optimal_C(y)

        kernel_matrices = [k1, k2]

        d_init = np.ones(len(kernel_matrices))/len(kernel_matrices)
        d, combined, J, alpha, gap = mkl_svdd.find_kernel_weights(
            d_init, kernel_matrices, C=C, y=np.zeros(len(data)), verbose=False)

        d = fix_d(d, cutoff=self.cutoff)

        return get_combined_kernel(kernel_matrices, d)

    def get_name(self):
        return "mkl-" + self.ex1.get_name() + self.ex2.get_name() + str(self.cutoff)
