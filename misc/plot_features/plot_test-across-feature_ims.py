"""
To get an intuitive idea on the trend for failed bearings, e.g. degradation
With the produced degradation onset
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import os
from icecream import ic

from util import *
from vib_record_ims import VibRecordIms
from vib_predict import VibPredict

os.chdir('../..')


if __name__ == '__main__':
    rec = VibRecordIms()
    p = VibPredict()

    num_feat = config('ims.num_features')
    onsets = config('ims.prev_onsets')
    for idx_tst in range(rec.NUM_TST):
        idxs_brg = rec.BRGS_FLD[idx_tst]
        for m, idx_brg in enumerate(idxs_brg):
            fig, axs = plt.subplots(num_feat, figsize=(16, 12), constrained_layout=True)
            for idx, (feat, feat_disp) in enumerate(rec.FEAT_DISP_NMS.items()):
                x = rec.get_time_axis(idx_tst=idx_tst)
                y = rec.get_feature_series(idx_tst, idx_brg, feat=feat)
                idx_tst_onset = max(0, idx_tst-1)
                onset = onsets[str(idx_tst_onset)][str(idx_brg)]

                axs[idx].plot(x, y, marker='o', markersize=0.5, lw=0.125, label='Observed values')
                axs[idx].xaxis.set_major_formatter(mdates.DateFormatter(rec.T_FMT))
                axs[idx].set_title(f'{feat_disp}, failed bearing {idx_brg+1}')
                axs[idx].axvline(x=x[onset], c='r', label='Detected degradation onset', lw=0.5)

            title = f'Trend for all features, run-to-failure-test {idx_tst + 1}, failed bearing {idx_brg+1}'
            plt.suptitle(title)
            ic(title)
            plt.savefig(f'plot/{title}', dpi=300)


