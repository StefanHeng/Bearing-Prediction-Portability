"""
Show as e.g., the degradation onset for previous DIs that failed in the FEMTO dataset
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


def plot_single():
    feat_d = rec.FEAT_DISP_NMS[feat]
    x = rec.get_time_axis(idx_tst=idx_tst)
    y = rec.get_feature_series(idx_tst, feat=feat)  # For getting the 1st channel
    onset_t = onsets_truth[idx_tst]
    onset_d = onsets_detected_prev[str(idx_tst)][feat]
    if onset_d != -1:
        axes[idx_plt].axvline(x=x[onset_d], color='r', label='Degradation onset detected', lw=0.5)
    axes[idx_plt].axvline(x=x[onset_t], color='g', label='Degradation onset manual ground truth', lw=0.5)
    axes[idx_plt].plot(x, y, marker='o', markersize=0.5, lw=0.125, label=f'Observed trend')
    axes[idx_plt].xaxis.set_major_formatter(mdates.DateFormatter(rec.T_FMT))
    axes[idx_plt].legend()
    axes[idx_plt].set_title(f'Run-to-failure test {rec.tst_nm(idx_tst)}: {feat_d} over time')
    axes[idx_plt].set_xlabel('time (dd hh:mm)')


if __name__ == '__main__':
    rec = VibRecordFemto()
    p = VibPredict()
    onsets_truth = config('femto.onset_truth')
    onsets_detected_prev = config('femto.degrading_onsets_by_feature_prev')
    ic(onsets_truth, onsets_detected_prev)

    fig, axes = plt.subplots(2, figsize=(9, 6))

    idx_tst = 1
    feat = 'kurtosis'
    idx_plt = 0
    plot_single()

    idx_tst = 2
    feat = 'skewness'
    idx_plt = 1
    plot_single()
    # feat_d = rec.FEAT_DISP_NMS[feat]
    # x = rec.get_time_axis(idx_tst=idx_tst)
    # y = rec.get_feature_series(idx_tst, feat=feat)  # For getting the 1st channel
    # axes[1].plot(x, y, marker='o', markersize=0.5, lw=0.125, label=f'Observed trend')
    # axes[1].xaxis.set_major_formatter(mdates.DateFormatter(rec.T_FMT))
    # axes[1].legend()
    # axes[1].set_title(f'Run-to-failure test {rec.tst_nm(idx_tst)}: {feat_d} over time')
    # axes[1].set_xlabel('time (dd hh:mm)')

    title = f'Output of previous degrading indicators on the FEMTO dataset_the failed tests and features'
    # plt.suptitle(title)
    # plt.show()
    fig.tight_layout(pad=2)
    plt.savefig(f'present_final-report/{title}.png', dpi=300)
