"""
Show as e.g., the degradation onset for new DI that fits the FEMTO dataset
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


if __name__ == '__main__':
    rec = VibRecordFemto()
    p = VibPredict()
    onsets_truth = config('femto.onset_truth')
    onsets_detected_new = config('femto.degrading_onsets_by_feature_new')
    ic(onsets_truth, onsets_detected_new)

    fig, ax = plt.subplots(1, figsize=(9, 3))

    idx_tst = 3
    feat = 'mean_freq'
    feat_d = rec.FEAT_DISP_NMS[feat]
    x = rec.get_time_axis(idx_tst=idx_tst)
    y = rec.get_feature_series(idx_tst, feat=feat)  # For getting the 1st channel
    ax.plot(x, y, marker='o', markersize=0.5, lw=0.125, label=f'Observed trend')

    onset_t = onsets_truth[idx_tst]
    onset_d = onsets_detected_new[str(idx_tst)][feat]
    if onset_d != -1:
        ax.axvline(x=x[onset_d], color='r', label='Degradation onset detected', lw=0.5)
    ax.axvline(x=x[onset_t], color='g', label='Degradation onset manual ground truth', lw=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(rec.T_FMT))
    ax.legend()
    ax.set_title(f'Run-to-failure test {rec.tst_nm(idx_tst)}: {feat_d} over time')
    ax.set_xlabel('time (dd hh:mm)')

    title = f'Output of newly selected degrading indicators on the FEMTO dataset_Mean in frequency that fits'
    # plt.suptitle(title)
    # plt.show()
    # fig.tight_layout(pad=0)
    plt.savefig(f'present_final-report/{title}.png', dpi=300, bbox_inches='tight')

