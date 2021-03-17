import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy.stats import linregress

import matplotlib.pyplot as plt

from scipy.stats import pearsonr

from vib_record import VibRecord
from vib_transfer import VibTransfer

from icecream import ic

if __name__ == "__main__":
    rec = VibRecord()
    t = VibTransfer()
    # ic(t.degrad_detect_score(rec.get_feature_series(0), plot='RMS in time'))
    vals_time = rec.get_feature_series(0, feat='kurtosis')
    vals_time_n = vals_time / (np.max(vals_time) - np.min(vals_time))
    vals_freq = rec.get_feature_series(0, feat='mean_freq')
    vals_freq_n = vals_freq / (np.max(vals_freq) - np.min(vals_freq))

    N = vals_time.size
    x = np.arange(N)
    min_split = 30
    idxs = x[min_split:N - min_split + 1]

    m, b = linregress(x[:1000], vals_time_n[:1000])[:2]
    ic(t.rmse(x[:1000], vals_time_n[:1000], m, b))
    m, b = linregress(x[:1000], vals_freq_n[:1000])[:2]
    ic(t.rmse(x[:1000], vals_freq_n[:1000], m, b))

    corrs_h_t = np.absolute(np.vectorize(lambda idx: pearsonr(x[:idx], vals_time[:idx])[0])(idxs))
    corrs_d_t = np.absolute(np.vectorize(lambda idx: pearsonr(x[idx:], vals_time[idx:])[0])(idxs))
    # m_slopes_h = np.vectorize(lambda idx: polyfit(x[:idx], vals_time[:idx], 1)[0])(idxs)  # Needs normalized data

    r_sqr_h_t = (np.vectorize(lambda idx: linregress(x[:idx], vals_time_n[:idx])[2])(idxs)) ** 2
    r_sqr_d_t = (np.vectorize(lambda idx: linregress(x[idx:], vals_time_n[idx:])[2])(idxs)) ** 2

    corrs_h_f = np.absolute(np.vectorize(lambda idx: pearsonr(x[:idx], vals_freq[:idx])[0])(idxs))
    corrs_d_f = np.absolute(np.vectorize(lambda idx: pearsonr(x[idx:], vals_freq[idx:])[0])(idxs))

    r_sqr_h_f = (np.vectorize(lambda idx: linregress(x[:idx], vals_freq_n[:idx])[2])(idxs)) ** 2
    r_sqr_d_f = (np.vectorize(lambda idx: linregress(x[idx:], vals_freq_n[idx:])[2])(idxs)) ** 2

    plt.figure(figsize=(16, 9))
    plt.plot(x, vals_time_n, 'o', label='RMS, t', markersize=0.5)
    plt.plot(x, vals_freq_n, 'o', label='Mean, f', markersize=0.5)
    # plt.plot(idxs, corrs_h_t + corrs_d_t, label='Corr, t', linewidth=1)
    # # plt.plot(idxs, corrs_d_t, label='Corr degrading, t', linewidth=1)
    # plt.plot(idxs, corrs_h_f + corrs_d_f, label='Corr, f', linewidth=1)
    # # plt.plot(idxs, corrs_d_f, label='Corr degrading, f', linewidth=1)

    plt.plot(idxs, r_sqr_h_t + r_sqr_d_t, label='R2, t', linewidth=1)
    plt.plot(idxs, r_sqr_h_f + r_sqr_d_f, label='R2, f', linewidth=1)
    plt.legend(loc=0)
    plt.show()
