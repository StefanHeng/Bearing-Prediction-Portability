"""
Example plots for degradation indicator selection & algorithm output
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import os
from icecream import ic

from util import *
from vib_record_femto import VibRecordFemto
# from vib_record_ims import VibRecordIms
from vib_predict import VibPredict

os.chdir('..')


def single_plot():
    x = rec.get_time_axis(idx_tst=idx_tst)
    y = rec.get_feature_series(idx_tst, feat=feat)
    onset = onsets[str(idx_tst)][feat]

    fig, ax = plt.subplots(1, figsize=(9, 6), constrained_layout=True)
    plt.plot(x, y, marker='o', markersize=0.5, lw=0.125, label='Extracted values')
    if onset != -1:
        plt.axvline(x=x[onset], color='r', label='Degradation onset detected', lw=0.5)
    plt.axvline(x=x[truth[idx_tst]], color='g', label='Degradation onset ground truth', lw=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(rec.T_FMT))
    plt.legend()
    plt.suptitle(title)
    # plt.show()
    plt.savefig(f'present_final/{title}.png', dpi=300)


if __name__ == '__main__':
    rec = VibRecordFemto()
    p = VibPredict()

    onsets = config('femto.degrading_onsets_by_feature_prev')
    truth = config('femto.onset_truth')
    ic(onsets, truth)

    # Tried to tune original Degrading indicators, the failed tests & features
    idx_tst = 5
    feat = 'skewness'
    title = f'prev-DI_Degradation detected too early: `{rec.FEAT_DISP_NMS[feat]}` on run-to-failure test {idx_tst+1}'
    single_plot()
    idx_tst = 1
    feat = 'kurtosis'
    title = f'prev-DI_Unable to detect degradation: `{rec.FEAT_DISP_NMS[feat]}` on run-to-failure test {idx_tst+1}'
    single_plot()

    onsets = config('femto.degrading_onsets_by_feature_new')
    ic(onsets)

    idx_tst = 2
    feat = 'mean_freq'
    title = f'new-DI_A good degradation output: `{rec.FEAT_DISP_NMS[feat]}` on run-to-failure test {idx_tst+1}'
    single_plot()

    idx_tst = 4
    feat = 'range_time'
    title = f'new-DI_And understandable degradation output: ' \
            f'`{rec.FEAT_DISP_NMS[feat]}` on run-to-failure test {idx_tst+1}'
    single_plot()
