import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import os
from icecream import ic

from vib_record_femto import VibRecordFemto
from vib_predict import VibPredict

os.chdir('../..')


if __name__ == '__main__':
    rec = VibRecordFemto()
    p = VibPredict()

    # feat = 'skewness'
    feat = 'kurtosis'
    sz_window = 2

    fig, axs = plt.subplots(rec.NUM_BRG_TST, figsize=(16, 12), constrained_layout=True)
    for idx_brg in range(rec.NUM_BRG_TST):
        # x = rec.get_time_axis(idx_brg=idx_brg)
        y = rec.get_feature_series(idx_brg, feat=feat)
        x = np.arange(y.size)
        x_mean, mean = p.sliding_means(y, sz_window)

        axs[idx_brg].plot(x, y, marker='o', markersize=0.25, linewidth=0.125, label='ori')
        axs[idx_brg].plot(x_mean, mean, marker='o', markersize=0.25, linewidth=0.125, label='smoothed')
        axs[idx_brg].xaxis.set_major_formatter(mdates.DateFormatter(rec.T_FMT))

        axs[idx_brg].set_title(f'Test bearing {idx_brg + 1}')

    plt.legend()

    # plt.suptitle(title)
    # plt.savefig(f'plot/{title}', dpi=300)
    plt.show()
