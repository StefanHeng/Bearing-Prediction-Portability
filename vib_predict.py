import numpy as np

from math import tanh

from vib_transfer import VibTransfer

from icecream import ic


class VibPredict:
    """ The 2-fold prediction model with potential adaptations
     1st part: predict degradation onset
     2nd part: predict failure time/Remaining Useful Life(RUL) """

    # Default initial number of measurements to compute baseline for degradation onset, as a fraction of input data size
    FRAC_SZ_BASE = 1 / 5
    MIN_SZ_BASE = 100

    @staticmethod
    def degradation_onset_prev(series, sz_base=100, sz_window=30, z=3):
        """
        Degradation onset defined by the first devidation of mean of a moving window starting from sz_base, from
        the range (mean ± z * std) of all previous data, staring from initial batch of size sz_base

        :param series: numpy 1D Array of health indicator
        :param sz_base: Initial number of measurements to computed baseline, normal range
        :param sz_window: Size of the window to compute mean for checking out of range
        :param z: The number of std away from mean to compute indicator range considered healthy
        :return: The index relative to input measurement array for degradation onset, -1 if degradation onset not found

        .. note:: Assumes degradation would occur
        """
        for idx in range(sz_base, series.size - 1):  # Array at the end would have size of 1
            m_prev = series[:idx].mean()
            std_prev = series[:idx].std()
            m_wd = series[idx:idx + sz_window].mean()
            if not (m_prev - z * std_prev <= m_wd <= m_prev + z * std_prev):
                return idx
        return -1

    @staticmethod
    def degradation_onset_dynamic(series, frac_sz_base=FRAC_SZ_BASE, min_sz_base=MIN_SZ_BASE, sz_window=30, z=3):
        sz_base = int(max(series.size * frac_sz_base, min_sz_base))
        # z = VibPredict._dynamic_z_limit(series, frac_sz_base, min_sz_base)
        # ic(z)
        return VibPredict.degradation_onset_prev(series, sz_base=sz_base, sz_window=sz_window, z=z)

    @staticmethod
    def _dynamic_z_limit(series, frac_sz_base=FRAC_SZ_BASE, min_sz_base=MIN_SZ_BASE):
        """
        To adapt previous degradation onset detection to generalize well to unseen dataset.

        Adjust hyper parameter, z, multiplier for standard deviation based healthy range

        A mapping from higher normalized RMSE of series[:frac_sz_base] to lower z

        Through hinge loss

        :param series: Health indicator data series
        :param frac_sz_base: Initial number of measurements to computed baseline, normal range,
        as a fraction of input data size
        :param min_sz_base: Minimal base size
        :return: Recommended multiplier z for degradation onset detection
        """
        sz_base = int(max(series.size * frac_sz_base, min_sz_base))
        s = series[:sz_base]
        # x = VibTransfer.rmse((s - s.mean()) / s.std())
        x = abs(s.std() / s.mean())
        ic(x)
        return 3 - 1.5 * tanh(max(x - 1, 0))

    @staticmethod
    def degradation_onset_normal(series, sz_base=100, sz_window=50, z=3, p=0.75):
        """
        Degrading point is defined by the number of points outside of multiplier range in the window
        """
        for idx in range(sz_base, series.size - 1):  # Array at the end would have size of 1
            m_prev = series[:idx].mean()
            std_prev = series[:idx].std()
            w = series[idx:idx + sz_window]
            count = ((w <= m_prev - z * std_prev) | (w >= m_prev + z * std_prev)).sum()
            if count >= sz_window * (1 - p):
                return idx
        return -1

    @staticmethod
    def degradation_onset_normal_weighted(series, sz_base=100, sz_window=50, z=3, t=10):
        """
        Degrading point is defined by the number of points outside of multiplier range in the window
        """
        for idx in range(sz_base, series.size - 1):  # Array at the end would have size of 1
            m_prev = series[:idx].mean()
            std_prev = series[:idx].std()
            w = (series[idx:idx + sz_window] - m_prev) / std_prev  # Window of z values for each data point
            idxs = np.concatenate([np.where(w < -z)[0], np.where(w > z)[0]])
            deviations = np.abs(w[idxs])

            if deviations.size > 0 and deviations.mean() >= t:
                return idx
        return -1

    @staticmethod
    def degradation_onset_normal_continuous_squash(series, sz_base=100, sz_window=50, z=3, t=0.75, toler=2):
        """
        Degrading point is defined by the number of points outside of multiplier range in the window
        """
        for idx in range(sz_base, series.size - 1):  # Array at the end would have size of 1
            m_prev = series[:idx].mean()
            std_prev = series[:idx].std()
            w = (series[idx:idx + sz_window] - m_prev) / std_prev  # Window of z values for each data point
            idxs = np.concatenate([np.where(w < -z)[0], np.where(w > z)[0]])
            deviations = np.abs(w[idxs])
            deviations = np.tanh(deviations / (z * toler))

            if deviations.size > 0 and deviations.mean() >= t:
                return idx
        return -1

    @staticmethod
    def degradation_onset_normal_dynamic(series, frac_sz_base=FRAC_SZ_BASE, min_sz_base=MIN_SZ_BASE, sz_window=30,
                                         z=3, p=0.75):
        sz_base = int(max(series.size * frac_sz_base, min_sz_base))
        return VibPredict.degradation_onset_normal(series, sz_base, sz_window, z, p)

    @staticmethod
    def degradation_onset_normal_dynamic_mean(series, frac_sz_base=FRAC_SZ_BASE, min_sz_base=MIN_SZ_BASE, sz_window=30,
                                              z=3, p=0.75, sz_mean=2):
        sz_base = int(max(series.size * frac_sz_base, min_sz_base))
        _, series = VibPredict.sliding_means(series, sz_mean)
        onset = VibPredict.degradation_onset_normal(series, sz_base, sz_window, z, p)
        return onset + sz_mean - 1 if onset != -1 else -1

    @staticmethod
    def degradation_onset_normal_weighted_dynamic_mean(series, frac_sz_base=FRAC_SZ_BASE, min_sz_base=MIN_SZ_BASE,
                                                       sz_window=30,
                                                       z=2, t=5):
        sz_base = int(max(series.size * frac_sz_base, min_sz_base))
        _, series = VibPredict.sliding_means(series, 4)
        return VibPredict.degradation_onset_normal_weighted(series, sz_base, sz_window, z, t)

    @staticmethod
    def degradation_onset_normal_continuous_squash_mean(series, frac_sz_base=FRAC_SZ_BASE, min_sz_base=MIN_SZ_BASE,
                                                        sz_window=30,
                                                        z=2, t=0.75, toler=2):
        sz_base = int(max(series.size * frac_sz_base, min_sz_base))
        _, series = VibPredict.sliding_means(series, 4)
        return VibPredict.degradation_onset_normal_continuous_squash(series, sz_base, sz_window, z, t, toler)

    @staticmethod
    def sliding_means(arr, sz_window=30):
        """ Mean is computed `sz_window` before all valid indices """
        idxs = np.arange(sz_window, arr.size - 1)
        return idxs, np.vectorize(lambda idx: arr[idx - sz_window:idx].mean())(idxs)
