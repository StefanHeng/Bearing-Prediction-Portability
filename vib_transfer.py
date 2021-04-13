import numpy as np

from math import tanh
from scipy.stats import linregress, shapiro, normaltest, anderson, norm
from scipy.stats import kurtosis as kurt, skew
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from unidip import UniDip

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot
from statsmodels.graphics.gofplots import qqplot
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
    def degrading_detection_fitness2(data, min_split_sz=MIN_SPILT_SZ):
        data = (data - data.mean()) / data.std()
        N = data.size
        x = np.arange(N)
        idxs = x[min_split_sz:N - min_split_sz + 1]

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

    @staticmethod
    def normality_ShapiroWilk(vals, alpha=0.05):
        """ Returns true if the data is considered normal by Shapiro-Wilk Test
        """
        stat, p = shapiro(vals)
        # ic(p)
        return p > alpha

    @staticmethod
    def normality_DAgostinoK2(vals, alpha=0.05):
        stat, p = normaltest(vals)
        # ic(p)
        return p > alpha

    @staticmethod
    def normality_AndersonDarling(vals, sign_lv=0.05):
        statistic, crit_vals, sign_lvs = anderson(vals)
        crit_vals: np.ndarray
        sign_lvs: np.ndarray
        idx = list(sign_lvs).index(sign_lv * 100)  # Get the corresponding percentage
        # ic(statistic, crit_vals[idx], sign_lvs[idx])
        return statistic < sign_lvs[idx]

    @staticmethod
    def normality_visual(vals, title='', save=False):
        """ Shows the histogram and Q-Q plot """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        # plt.hist(vals, ax=axes[0])
        # qqplot(vals, line='s', marker='o', markersize=0.5, plot=axes[1])

        # sns.distplot(vals, ax=axes[0])
        axes[0].hist(vals, bins=20, density=True, alpha=0.6)
        mu, std = norm.fit(vals)
        x = np.linspace(mu - 3 * std, mu + 3 * std, 100)
        axes[0].plot(x, norm.pdf(x, mu, std), c='r', lw=1)
        axes[0].set_title('Histogram with normal distribution fit')

        probplot(vals, plot=axes[1])
        lines = axes[1].get_lines()[0]
        lines.set_marker('o')
        lines.set_markerfacecolor('r')
        lines.set_markersize(0.5)
        lines.set_linewidth(0.125)
        axes[1].set_title('Q-Q plot')

        # plt.subplot(1, 2, 1)
        # plt.hist(vals)
        # plt.subplot(1, 2, 2)
        # qqplot(vals, line='s', marker='o', markersize=0.5)
        plt.suptitle(title)
        if save:
            plt.savefig(f'plot/{title}')
        else:
            plt.show()

    @staticmethod
    def trend_normal_enough(vals, alpha_m=0.05, t_s=4, z=None):
        """
        Checks if the data is symmetric and uni-modal

        .. note::

        :param vals: 1 dimensional data
        :param alpha_m: Confidence required to be a mode, as opposed to outlier, by Hartigan's dip test for modality
        :param t_s: Threshold for absolute value of skewness
        :param z: If z is specified as standard deviation multiplier, outliers on both extremes are removed
        """
        # k = kurt(vals)
        num_interval = len(UniDip(vals, alpha=alpha_m, ntrials=10).run())
        if z is not None:
            m = vals.mean()
            rang = vals.std() * z
            within_range = (m - rang < vals) & (vals < m + rang)
            size_ori = vals.size
            vals = vals[within_range]
            assert size_ori - vals.size <= size_ori * 0.03  # Make sure only a small fraction of outliers are removed
        s = skew(vals)
        return num_interval == 1 and abs(s) < t_s

    @staticmethod
    def evaluate_degradation_onset(tst_sz, onset, r=0.13):
        """
        # :return: Whether onset index is within range of starting and ending ratios relative to length of test
        :returnL Whether the degrading stage has long enough data by ratio in [0, 1]
        """
        # mid = tst_sz / 2
        # mid_rang = tst_sz * r_mid
        # in_mid_rang = mid - mid_rang <= onset <= mid + mid_rang
        # long_enough_degrad = (tst_sz - onset) >= tst_sz * r_d
        # if not verbose:
        #     return in_mid_rang and long_enough_degrad
        # else:
        #     return in_mid_rang, long_enough_degrad

        # r_s, r_e = r
        # ic(tst_sz * r_s, onset, tst_sz * r_e)
        # return tst_sz * r_s <= onset <= tst_sz * r_e
        return tst_sz - onset > tst_sz * r

