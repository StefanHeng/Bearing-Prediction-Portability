import numpy as np
from scipy.stats import pearsonr, kendalltau

import matplotlib.pyplot as plt
from icecream import ic


class VibTransfer:
    """ Handles transferring a vibration prediction model to a new dataset

    Metrics to evaluate what to transfer
    """

    @staticmethod
    def degrading_diff(data, n=5, C=(0.5, 0.5, 0.25)):
        """
        Evaluates how good is a feature to differentiate degradation onset

        :param data: A bearing degradation trend
        :param n: Number of maximal metrics to average over
        :param C: Multiplicative factors for each term
        :return: Average of max n (Difference between correlation of a split point +
        Difference between rank correlation of a split point +
        C * the fraction of the split point by size of `data`)

        .. note:: `n` and `C` are hyper-parameters for response to outliers and relative importance
        """
        N = data.size
        x = np.arange(N)
        idxs = x[2:N-2+1]  # All valid split indices to ensure 2 data points minimum
        corrs = np.vectorize(lambda idx: VibTransfer._degrading_diff(x, data, idx, pearsonr))(idxs)
        taus = np.vectorize(lambda idx: VibTransfer._degrading_diff(x, data, idx, kendalltau))(idxs)
        ic(np.stack((corrs, taus, idxs/N), axis=-1))
        metrics = np.matmul(np.stack((corrs, taus, idxs/N), axis=-1), np.array(C))
        ic(metrics)
        plt.plot(idxs, corrs, label='r')
        plt.plot(idxs, taus, label='tau')
        plt.plot(idxs, metrics, label='metric')
        plt.legend(loc=0)
        plt.show()

    @staticmethod
    def _degrading_diff(x, y, idx, func):
        return abs(func(x[idx:], y[idx:])[0] - func(x[:idx], y[:idx])[0])  # 1st of the tuple is the correlation

