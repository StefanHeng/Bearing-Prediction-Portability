import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import os
from icecream import ic

from vib_record_femto import VibRecordFemto
# from vib_record_ims import VibRecordIms
from vib_predict import VibPredict
from vib_transfer import VibTransfer


os.chdir('../..')


if __name__ == '__main__':
    rec = VibRecordFemto()
    p = VibPredict()
    t = VibTransfer()

    series_good_strt_trend = rec.get_feature_series(0, feat='rms_time')
    series_bad_strt_trend = rec.get_feature_series(1, feat='kurtosis')

    sz_g = series_good_strt_trend.size
    s_g = series_good_strt_trend[:sz_g // 5]
    s_g = (s_g - s_g.mean()) / s_g.std()

    sz_b = series_bad_strt_trend.size
    s_b = series_good_strt_trend[:sz_b // 5]
    s_b = (s_b - s_b.mean()) / s_b.std()

    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(s_g.size), s_g, marker='o', markersize=1, linewidth=0.25)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(s_b.size), s_b, marker='o', markersize=1, linewidth=0.25)
    plt.show()

    ic(
        t.rmse(s_g),
        t.rmse(s_b)
    )
