"""
Show a degenerate trend where the degradation detection fails
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import os
from icecream import ic

from util import *
from vib_record_femto import VibRecordFemto
# from vib_record_ims import VibRecordIms
from vib_predict import VibPredict

os.chdir('..')


def prev_mean_n_std(arr, sz_base=100):
    """ Incremental mean & standard deviation of all values seen before i, for i starting from `sz_base` """
    idxs = np.arange(sz_base, arr.size - 1)
    return (
        idxs,
        np.vectorize(lambda idx: arr[:idx].mean())(idxs),
        np.vectorize(lambda idx: arr[:idx].std())(idxs)
    )


if __name__ == '__main__':
    rec = VibRecordFemto()
    p = VibPredict()
    onsets_truth = config('femto.onset_truth')
    onsets_detected_new = config('femto.degrading_onsets_by_feature_new')
    params = config('femto.degrading_hyperparameters_new')
    ic(onsets_truth, onsets_detected_new, params)

    fig, ax = plt.subplots(1, figsize=(9, 3))

    idx_tst = 1
    feat = 'range_time'
    feat_d = rec.FEAT_DISP_NMS[feat]
    x = rec.get_time_axis(idx_tst=idx_tst)
    y = rec.get_feature_series(idx_tst, feat=feat)  # For getting the 1st channel
    sz_base = params[feat]['sz_base']
    idxs, means = p.sliding_means(y, sz_window=params[feat]['sz_window'], strt=sz_base)
    idxs_prev, prev_means, prev_stds = prev_mean_n_std(y, sz_base=sz_base)
    prev_me = prev_stds * params[feat]['z']

    onset_t = onsets_truth[idx_tst]
    onset_d = onsets_detected_new[str(idx_tst)][feat]
    if onset_d != -1:
        ax.axvline(x=x[onset_d], color='r', label='Degradation onset detected', lw=0.5)
    ax.axvline(x=x[onset_t], color='g', label='Degradation onset manual ground truth', lw=0.5)

    ax.plot(x, y, marker='o', markersize=0.25, lw=0.125, label=f'Observed trend')
    ax.plot(x[idxs], means, c='purple', lw=0.5, label=f'Mean of moving window')
    ax.plot(x[idxs_prev], prev_means, c='orange', lw=0.5, label=r'$[\bar{x} \pm zs]$ for historical values')
    ax.fill_between(x[idxs_prev], prev_means-prev_me, prev_means+prev_me, facecolor='orange', alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(rec.T_FMT))
    ax.legend()
    ax.set_title(f'Run-to-failure test {rec.tst_nm(idx_tst)}: {feat_d} over time,'
                 f' with sliding degradation detection range')
    ax.set_xlabel('time (dd hh:mm)')

    title = f'Degenerate case on the FEMTO dataset_Range in time'
    # plt.suptitle(title)
    # plt.show()
    # fig.tight_layout(pad=0)
    plt.savefig(f'present_final-report/{title}.png', dpi=300, bbox_inches='tight')

