import numpy as np
from scipy.stats import pearsonr
from numpy.polynomial.polynomial import polyfit
from math import tanh

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from icecream import ic


class VibTransfer:
    """ Handles transferring a vibration prediction model to a new dataset

    Metrics to evaluate what to transfer
    """

    MIN_SPILT_SZ = 30

    def __init__(self, min_split_sz=MIN_SPILT_SZ):
        self.min_split_sz = min_split_sz

    def degrad_detect_score(self, data, n=5, C=np.full(5, 1 / 5), plot=''):
        """
        Evaluates how good is a feature to differentiate degradation onset

        :param data: A bearing degradation trend
        :param n: Number of maximal metrics to average over
        :param C: Multiplicative factors for each term
        :param plot: If specified, the terms and the final metric would be exported
        :return: Average of max `n` (Correlation before split point +
        How close to 0 is slope before split point +
        Correlation after split point +
        the fraction of the split point by size of `data`) weighted by `C`

        .. note:: `n` and `C` are hyper-parameters for response to outliers and relative importance
        """
        C = np.array(C)
        data_n = data / (np.max(data) - np.min(data))  # For fair comparison across features
        N = data.size
        x = np.arange(N)
        idxs = x[self.MIN_SPILT_SZ:N - self.MIN_SPILT_SZ + 1]

        # def _degrad_detect_score(idx):  # Stacking is faster
        #     x_h, y_h = x[:idx], data[:idx]
        #     corr_h = pearsonr(x_h, y_h)[0]  # `pearsonr` returns (correlation, p-value)
        #     slope_h = polyfit(x_h, y_h, 1)[0]
        #     corr_d = pearsonr(x[idx:], data[idx:])[0]
        #     terms = [
        #         corr_h,
        #         1 - tanh(abs(slope_h)),
        #         corr_d,
        #         idx / N
        #     ]
        #     return np.dot(np.array(terms), C)
        #
        # metrics = np.vectorize(lambda idx: _degrad_detect_score(idx))(idxs)

        corrs_h = np.vectorize(lambda idx: pearsonr(x[:idx], data[:idx])[0])(idxs)
        m_corrs_h = np.absolute(corrs_h)
        slopes_h = np.vectorize(lambda idx: polyfit(x[:idx], data_n[:idx], 1)[0])(idxs)  # Needs normalized data
        m_slopes_h = 1 - np.tanh(np.absolute(slopes_h))

        corrs_d = np.vectorize(lambda idx: pearsonr(x[idx:], data[idx:])[0])(idxs)
        m_corrs_d = np.absolute(corrs_d)
        slopes_d = np.vectorize(lambda idx: polyfit(x[idx:], data_n[idx:], 1)[0])(idxs)
        m_slopes_d = np.tanh(np.absolute(slopes_d))

        metrics = np.matmul(np.stack((m_corrs_h, m_slopes_h, m_corrs_d, m_slopes_d, idxs / N), axis=-1), np.array(C))
        idxs_max = np.argpartition(metrics, -n)[-n:]

        if plot:
            plt.figure()
            plt.plot(x, data_n, 'o', label='Data, normalized', markersize=0.5)
            plt.plot(idxs, m_corrs_h, label='Correlation before split', linewidth=0.5)
            plt.plot(idxs, m_slopes_h, label='How close to 0 is slope before split', linewidth=0.5)
            plt.plot(idxs, m_corrs_d, label='Correlation after split', linewidth=0.5)
            plt.plot(idxs, idxs / N, label='Split fraction over duration', linewidth=0.5)
            plt.plot(idxs, metrics, label='Score', linewidth=1)
            plt.legend(loc=0)
            plt.suptitle(plot)
            fdr = 'plot/03.16.21 Degradation onset detection feature fitness/'
            plt.savefig(f'{fdr}Degradation onset detection feature fitness, {plot}.png', dpi=300)
            plt.close()
        return metrics[idxs_max].mean()

    @staticmethod
    def rmse(x, y, b, m):
        """ Root mean squared error for linear regression """
        return np.sqrt(mean_squared_error(y, m * x + b))
