import numpy as np
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis as kurt, skew
from scipy.signal import periodogram

# from icecream import ic

from femto_attrs import *


class VibExtract:
    """ Feature extraction function library on vibration signals

    Given vibration samples of `np.ndarray`, VibExtr supports the following properties on the first axis,
    with returned element/array in the subsequent dimensions.

    In time domain on amplitude: RMS, Range, Kurtosis, Skewness

    In frequency domain: RMS, Frequency with max amplitude, Mean, Rotating frequency
    """

    def __init__(self, num_data=N_SPL, fqs=SPL_RT):
        """
        :param num_data: Number of observations for a single data file
        """
        self.N_DATA = num_data
        self.FQS = fqs
        self.D_PROP_FUNC = {  # Dictionary of all functions on property
            'RMS in time': self.rms_time,
            'Range in time': self.range_time,
            'Kurtosis in time': self.kurtosis,
            'Skewness in time': self.skewness,
            'RMS in frequency': self.rms_freq,
            'Mean in frequency': self.mean_freq,
            'Max in frequency': self.peak_freq
        }

    @staticmethod
    def rms_time(vals):
        return np.sqrt(np.mean(np.square(vals), axis=0))

    @staticmethod
    def range_time(vals):
        return np.amax(vals, axis=0) - np.amin(vals, axis=0)

    @staticmethod
    def kurtosis(vals):
        return kurt(vals, axis=0, fisher=False)

    @staticmethod
    def skewness(vals):
        return skew(vals, axis=0)

    def rms_freq(self, vals):
        P1 = self._get_P1(vals)
        # ic(P1, P1.shape)
        freqs = fftfreq(self.N_DATA, d=1 / self.FQS)[:self.N_DATA // 2]
        if len(vals.shape) > 1:
            freqs = freqs.reshape((-1, 1))
        sums_P1 = np.sum(P1, axis=0)
        RMS = np.sqrt(np.sum(np.square(freqs) * P1, axis=0) / sums_P1)
        return RMS

    def _get_P1(self, vals):
        Y = fft(vals, self.N_DATA, axis=0)
        # ic(Y, Y.shape)
        P2 = np.abs(Y / self.N_DATA)
        return P2[:self.N_DATA // 2] * 2

    def mean_freq(self, vals):
        # TODO, still different from MATLAB native function
        if len(vals.shape) == 1:  # 1D array
            return self._mean_freq(vals)
        else:
            return np.apply_along_axis(self._mean_freq, 0, vals)

    def _mean_freq(self, arr):
        f, Pxx_den = periodogram(arr, self.FQS)
        return np.dot(f, Pxx_den) / np.sum(Pxx_den)

    def peak_freq(self, vals):
        return np.argmax(self._get_P1(vals), axis=0)
