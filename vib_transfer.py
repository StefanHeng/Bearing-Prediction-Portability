import numpy as np
from scipy.stats import pearsonr, kendalltau

import matplotlib.pyplot as plt
from icecream import ic


class VibTransfer:
    """ Handles transferring a vibration prediction model to a new dataset

    Metrics to evaluate what to transfer
    """

    @staticmethod
    def degrad_diff(data, n=5, C=(0.25, 0.5, 0.25)):
        """
        Evaluates how good is a feature to differentiate degradation onset

        :param data: A bearing degradation trend
        :param n: Number of maximal metrics to average over
        :param C: Multiplicative factors for each term
        :return: Average of max n (Difference of mean at a split point +
        Difference of standard deviation at split point +
        C * the fraction of the split point by size of `data`)

        .. note:: `n` and `C` are hyper-parameters for response to outliers and relative importance
        """
        data = (data - data.mean()) / data.std()
        N = data.size
        x = np.arange(N)
        idxs = x[2:N-2+1]  # All valid split indices to ensure 2 data points minimum
        mean = np.vectorize(lambda idx: VibTransfer._diff_by_split(data, idx, np.mean))(idxs)
        std = np.vectorize(lambda idx: VibTransfer._diff_by_split(data, idx, np.std))(idxs)
        metrics = np.matmul(np.stack((mean, std, idxs/N), axis=-1), np.array(C))
        idxs_max = np.argpartition(metrics, -n)[-n:]
        # plt.plot(idxs, mean, label='mean')
        # plt.plot(idxs, std, label='std')
        # plt.plot(idxs, metrics, label='metric')
        # plt.legend(loc=0)
        # plt.show()
        return metrics[idxs_max].mean()

    @staticmethod
    def _diff_by_split(data, idx, func):
        """ Computes unsigned difference of a metric by a split index """
        return abs(func(data[idx:]) - func(data[:idx]))

