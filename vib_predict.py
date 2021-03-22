import numpy as np


class VibPredict:
    """ The 2-fold prediction model with potential adaptations
     1st part: predict degradation onset
     2nd part: predict failure time/Remaining Useful Life(RUL) """

    @staticmethod
    def degradation_onset(series, sz_base=100, sz_window=30, z=3):
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
        n = series.size
        for idx in range(sz_base, n-1):  # Array at the end would have size of 1
            m_prev = series[:sz_base].mean()
            std_prev = series[:sz_base].std()
            # m_wd =



