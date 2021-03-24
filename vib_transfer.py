import numpy as np

from math import tanh
from scipy.stats import linregress
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

    @staticmethod
    def degrading_detection_fitness(data, n=5, C=np.full(NUM_TERM, 1 / NUM_TERM), min_split_sz=MIN_SPILT_SZ,
                                    plot=None):
        """
        Evaluates how good is a feature to differentiate degradation onset

        :param data: A bearing degradation trend
        :param n: Number of maximal metrics to average over
        :param C: Multiplicative factors for each term
        :param min_split_sz: Minimal number of data points for one partition, to prevent skewed values in the extreme cases
        :param plot: If specified, the terms and the final metric would be exported
        :return: Average of max `n` (Correlation before split point +
        How close to 0 is slope before split point +
        Correlation after split point +
        the fraction of the split point by size of `data`) weighted by `C`

        .. note:: `n` and `C` are hyper-parameters for response to outliers and relative importance
        """
        C = np.array(C)
        data = (data - data.mean()) / data.std()
        N = data.size
        x = np.arange(N)
        idxs = x[min_split_sz:N - min_split_sz + 1]

        def _degrading_detection_fitness(idx):
            x_h, y_h = x[:idx], data[:idx]
            m_h, b_h, _r_h, p_h = linregress(x_h, y_h)[:4]  # p-value
            x_d, y_d = x[idx:], data[idx:]
            m_d, b_d, _r_d, p_d = linregress(x_d, y_d)[:4]
            terms = [
                1 - tanh(VibTransfer._rmse(x_h, y_h, m_h, b_h)),  # Prefer smaller values
                p_h,
                1 - p_d,
                idx / N
            ]
            return np.array(terms)

        # Option 1: Normal index mapping
        # metrics = np.zeros((idxs.size, self.NUM_TERM))
        # for i in range(idxs.size):
        #     metrics[i] = _degrading_detection_fitness(i+self.MIN_SPILT_SZ)

        # Option 2: Hack to apply `np.apply_along_axis`, element into 1-array
        metrics = np.apply_along_axis(lambda idx: _degrading_detection_fitness(*idx), -1, idxs.reshape(-1, 1))

        metrics_s = metrics @ C
        idxs_max = np.argpartition(metrics_s, -n)[-n:]

        if plot:
            plt.figure(figsize=(16, 9), constrained_layout=True)
            plt.plot(x, data / (np.max(data) - np.min(data)), marker='o', label='Data, normalized',
                     markersize=0.5, linewidth=0.125)
            plt.plot(idxs, metrics[:, 0], label='How consistent are values in healthy stage', linewidth=1)
            plt.plot(idxs, metrics[:, 1], label='Portability of slope of 0 in healthy stage', linewidth=1)
            plt.plot(idxs, metrics[:, 2], label='Portability of slope of non-0 in degrading stage', linewidth=1)
            plt.plot(idxs, metrics[:, 3], label='Fraction of test duration', linewidth=1)
            plt.plot(idxs, metrics_s, label='Final score', linewidth=2)
            plt.legend(loc=0)
            plt.suptitle(plot)
            plt.savefig(f'plot/{plot}.png', dpi=300)
            # plt.show()
            plt.close()
        return metrics[idxs_max].mean()

    @staticmethod
    def _rmse(x, y, m, b):
        """ Root mean squared error """
        return np.sqrt(mean_squared_error(y, m * x + b))

    @staticmethod
    def rmse(vals):
        """ RMSE against best fit line """
        x = np.arange(vals.size)
        m, b = linregress(x, vals)[:2]
        return VibTransfer._rmse(x, vals, m, b)

    @staticmethod
    def mape(x, y, m, b):
        return mean_absolute_percentage_error(y, m * x + b)
