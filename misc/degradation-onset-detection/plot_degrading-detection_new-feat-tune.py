"""
Pick my own degrading indicators based on previous criteria,
still, try to achieve a good degrading output on new dataset by tuning
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import os
from icecream import ic

from util import *
from vib_record_femto import VibRecordFemto
# from vib_record_ims import VibRecordIms
from vib_predict import VibPredict

os.chdir('../..')

if __name__ == '__main__':
    rec = VibRecordFemto()
    p = VibPredict()

    indicators_degrading_new = ['range_time', 'mean_freq', 'peak_freq']
    truth = config('femto.onset_truth')  # As a rough guidance
    params = dict(  # TODO: these tuning are discarded after reflection, treat those in config file as the final params
        range_time=dict(sz_base=60, sz_window=25, z=2),
        # kurtosis=dict(sz_base=60, sz_window=15, z=3),
        mean_freq=dict(sz_base=50, sz_window=30, z=2),
        peak_freq=dict(sz_base=50, sz_window=50, z=1)
    )
    onsets = dict()

    ic(indicators_degrading_new, truth)
    for idx_tst in range(rec.NUM_BRG_TST):
        fig, axs = plt.subplots(len(indicators_degrading_new), figsize=(16, 12), constrained_layout=True)
        for idx, feat in enumerate(indicators_degrading_new):
            feat_disp = rec.FEAT_DISP_NMS[feat]
            x = rec.get_time_axis(idx_tst=idx_tst)
            y = rec.get_feature_series(idx_tst, feat=feat)
            axs[idx].plot(x, y, marker='o', markersize=0.5, lw=0.125)

            onset = p.degradation_onset_prev_(y, **params[feat])
            if onset != -1:
                axs[idx].axvline(x=x[onset], color='r', label='Degradation onset detected', lw=0.5)
            axs[idx].axvline(x=x[truth[idx_tst]], color='g', label='Degradation onset ground truth', lw=0.5)

            axs[idx].xaxis.set_major_formatter(mdates.DateFormatter(rec.T_FMT))
            axs[idx].set_title(feat_disp)

            ic(idx_tst, feat, onset)
            if idx_tst not in onsets:
                onsets[idx_tst] = dict()
            onsets[idx_tst][feat] = onset
        plt.legend()

        title = f'FEMTO Degradation detection on new indicators, tuned hyper-parameter, test bearing {idx_tst + 1}'
        plt.suptitle(title)
        ic(title)
        plt.savefig(f'plot/{title}', dpi=300)
        # plt.show()

    ic(onsets)
