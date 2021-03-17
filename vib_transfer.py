import numpy as np
from scipy.stats import pearsonr
from numpy.polynomial.polynomial import polyfit
from scipy.stats import linregress
from math import tanh
from statistics import harmonic_mean

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import matplotlib.pyplot as plt
from icecream import ic


class VibTransfer:
    """ Handles transferring a vibration prediction model to a new dataset

    Metrics to evaluate what to transfer
    """

    MIN_SPILT_SZ = 30
    NORM_FACT = 100
    NUM_TERM = 4

    def __init__(self, min_split_sz=MIN_SPILT_SZ, norm_fact=NORM_FACT):
        self.min_split_sz = min_split_sz
        self.norm_fact = norm_fact

    def degrad_detect_score(self, data, n=5, C=np.full(NUM_TERM, 1 / NUM_TERM), plot=''):
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
        # data = data / (np.max(data) - np.min(data)) * self.norm_fact  # For fair comparison across features
        data = (data - data.mean()) / data.std()
        N = data.size
        # x = np.arange(N) / N * self.norm_fact
        x = np.arange(N)
        idxs = x[self.MIN_SPILT_SZ:N - self.MIN_SPILT_SZ + 1]

        def _degrad_detect_score(idx):  # Stacking is faster
            x_h, y_h = x[:idx], data[:idx]
            m_h, b_h, _r_h, p_h = linregress(x_h, y_h)[:4]  # p-value
            x_d, y_d = x[idx:], data[idx:]
            m_d, b_d, _r_d, p_d = linregress(x_d, y_d)[:4]
            terms = [
                1 - tanh(self.rmse(x_h, y_h, m_h, b_h)),  # Prefer smaller values
                # 1 - self.mape(x_h, y_h, m_h, b_h),
                # 1 - self.rmse(x_h, y_h, m_h, b_h),
                # 1 - self.rmse(x_h, y_h, m_h, b_h),

                # 1 - tanh(abs(m_h)),
                # tanh(p_h),
                p_h,

                # 1 - tanh(self.rmse(x_d, y_d, m_d, b_d)),
                # 1 - self.rmse(x_d, y_d, m_d, b_d),

                # tanh(abs(m_d)),
                # tanh(1 - p_d),
                1 - p_d,

                # min([
                #     1 - tanh(self.rmse(x_h, y_h, m_h, b_h)),
                #     p_h,
                # ]),
                # min([
                #     1 - tanh(self.rmse(x_d, y_d, m_d, b_d)),
                #     1 - p_d,
                # ]),
                idx / N
            ]
            return np.array(terms)
            # ic(terms)
            # return np.dot(np.array(terms), C)

        metrics = np.zeros((idxs.size, self.NUM_TERM))
        for i in range(idxs.size):
            metrics[i] = _degrad_detect_score(i+self.MIN_SPILT_SZ)

        # ic(metrics[:, 0])  # RMSEs

        # metrics = np.vectorize(lambda idx: _degrad_detect_score(idx))(idxs)

        # corrs_h = np.vectorize(lambda idx: pearsonr(x[:idx], data[:idx])[0])(idxs)
        # m_corrs_h = np.absolute(corrs_h)
        # slopes_h = np.vectorize(lambda idx: polyfit(x[:idx], data_n[:idx], 1)[0])(idxs)  # Needs normalized data
        # m_slopes_h = 1 - np.tanh(np.absolute(slopes_h))
        #
        # corrs_d = np.vectorize(lambda idx: pearsonr(x[idx:], data[idx:])[0])(idxs)
        # m_corrs_d = np.absolute(corrs_d)
        # slopes_d = np.vectorize(lambda idx: polyfit(x[idx:], data_n[idx:], 1)[0])(idxs)
        # m_slopes_d = np.tanh(np.absolute(slopes_d))
        #
        # metrics = np.matmul(np.stack((m_corrs_h, m_slopes_h, m_corrs_d, m_slopes_d, idxs / N), axis=-1), np.array(C))
        metrics_s = metrics @ C
        idxs_max = np.argpartition(metrics_s, -n)[-n:]

        # if plot:
        #     plt.figure(figsize=(16, 9), constrained_layout=True)
        #     plt.plot(x, data / (np.max(data) - np.min(data)), 'o', label='Data, normalized', markersize=0.5)
        #     plt.plot(idxs, metrics[:, 0], label='1 - tanh(RMSE), healthy', linewidth=1)
        #     plt.plot(idxs, metrics[:, 1], label='P[slope == 0], healthy', linewidth=1)
        #     # plt.plot(idxs, metrics[:, 2], label='RMSE, degrading', linewidth=0.5)
        #     plt.plot(idxs, metrics[:, 2], label='P[slope != 0], degrading', linewidth=1)
        #     plt.plot(idxs, metrics_s, label='Score', linewidth=2)
        #     plt.legend(loc=0)
        #     plt.suptitle(plot)
        #     # fdr = 'plot/03.16.21 Degradation onset detection feature fitness/'
        #     # plt.savefig(f'{fdr}Degradation onset detection feature fitness, {plot}.png', dpi=300)
        #     plt.savefig(f'{plot}.png', dpi=300)
        #     # plt.close()
        #     plt.show()
        return metrics[idxs_max].mean()

    @staticmethod
    def rmse(x, y, m, b):
        """ Root mean squared error """
        return np.sqrt(mean_squared_error(y, m * x + b))

    @staticmethod
    def mape(x, y, m, b):
        return mean_absolute_percentage_error(y, m * x + b)

    # @staticmethod
    # def _harmonic_mean(*vals):
    #     return len(vals) / sum([(1 / i) for i in vals])
