"""
Re-generate illustrative figures in previous project
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import os
from icecream import ic

from util import *
# from vib_record_femto import VibRecordFemto
from vib_record_ims import VibRecordIms
from vib_predict import VibPredict

os.chdir('..')


if __name__ == '__main__':
    rec = VibRecordIms()
    p = VibPredict()

    idx_tst = 0  # The 1st channel of test 1, as in previous project
    feat = 'kurtosis'
    feat_d = rec.FEAT_DISP_NMS[feat]

    fig, axes = plt.subplots(2, figsize=(9, 6))

    brg_stt = iter(['Healthy', 'Healthy', 'Failed', 'Failed'])
    clrs = iter(['dodgerblue', 'mediumseagreen', 'indianred', 'darkorange'])
    x = rec.get_time_axis(idx_tst=idx_tst)
    for idx_brg in [0, 1]:
        y = rec.get_feature_series(idx_tst, idx_brg, feat=feat)  # For getting the 1st channel
        axes[0].plot(x, y, marker='o', markersize=0.5, lw=0.125, c=next(clrs), label=f'Healthy bearing {idx_brg+1}')
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter(rec.T_FMT))
        axes[0].legend()
        axes[0].set_title(f'Healthy bearings for run-to-failure test {idx_tst+1}: {feat_d} over time')
        axes[0].set_xlabel('time (dd hh:mm)')
    for idx_brg in [2, 3]:
        y = rec.get_feature_series(idx_tst, idx_brg, feat=feat)  # For getting the 1st channel
        axes[1].plot(x, y, marker='o', markersize=0.5, lw=0.125, c=next(clrs), label=f'Failed bearing {idx_brg+1}')
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter(rec.T_FMT))
        axes[1].legend()
        axes[1].set_title(f'Failed bearings for run-to-failure test {idx_tst + 1}: {feat_d} over time')
        axes[1].set_xlabel('time (dd hh:mm)')

    title = f'IMS Selecting degrading indicators_Run-to-failure test {idx_tst+1} on kurtosis'
    # plt.suptitle(title)
    # plt.show()
    fig.tight_layout(pad=2)
    plt.savefig(f'present_final-report/{title}.png', dpi=300)
