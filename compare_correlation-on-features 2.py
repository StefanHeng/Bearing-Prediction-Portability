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
    k = 400
    # ic(t.degrad_detect_score(rec.get_feature_series(0), plot='RMS in time'))
    vals_time = rec.get_feature_series(0, feat='kurtosis')
    vals_time_n = vals_time / (np.max(vals_time) - np.min(vals_time)) * k
    # vals_time_n = (vals_time - vals_time.mean()) / vals_time.std()
    vals_freq = rec.get_feature_series(0, feat='mean_freq')
    vals_freq_n = vals_freq / (np.max(vals_freq) - np.min(vals_freq)) * k

    N = vals_time.size
    x = np.arange(N) / N
    min_split = 30
    idxs = x[min_split:N - min_split + 1]

    n = 1000
    m, b, r, p = linregress(x[:n], vals_time_n[:n])[:4]
    ic(m, b, p, t.rmse(x[:n], vals_time_n[:n], m, b))
    m, b, r, p = linregress(x[n:], vals_time_n[n:])[:4]
    ic(m, b, p, t.rmse(x[n:], vals_time_n[n:], m, b))

    m, b, r, p = linregress(x[:n], vals_freq_n[:n])[:4]
    ic(m, b, p, t.rmse(x[:n], vals_freq_n[:n], m, b))


    # rmse_t_h = t.rmse(x[:idx], data[:idx], *linregress(x_h, y_h)[:3])

    # plt.figure(figsize=(16, 9))
    # plt.plot(x, vals_time_n, 'o', label='RMS, t', markersize=0.5)
    # plt.plot(x, vals_freq_n, 'o', label='Mean, f', markersize=0.5)
    # #     # plt.plot(idxs, corrs_h_t + corrs_d_t, label='Corr, t', linewidth=1)
    # #     # # plt.plot(idxs, corrs_d_t, label='Corr degrading, t', linewidth=1)
    # #     # plt.plot(idxs, corrs_h_f + corrs_d_f, label='Corr, f', linewidth=1)
    # # # plt.plot(idxs, corrs_d_f, label='Corr degrading, f', linewidth=1)
    #
    # # plt.plot(idxs, r_sqr_h_t + r_sqr_d_t, label='R2, t', linewidth=1)
    # # plt.plot(idxs, r_sqr_h_f + r_sqr_d_f, label='R2, f', linewidth=1)
    # plt.legend(loc=0)
    # plt.show()
