"""
With degradation detection
"""


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import os
from icecream import ic

from util import *
from vib_record import VibRecord
from vib_record_ims import VibRecordIms
from vib_predict import VibPredict

os.chdir('../..')


if __name__ == '__main__':
    rec = VibRecord()
    p = VibPredict()

    for feat, feat_disp in rec.FEAT_DISP_NMS.items():
        fig, axs = plt.subplots(rec.NUM_BRG_TST, figsize=(16, 12), constrained_layout=True)
        for idx_tst in range(rec.NUM_BRG_TST):
            feat_disp = rec.FEAT_DISP_NMS[feat]
            x = rec.get_time_axis(idx_brg=idx_tst)
            y = rec.get_feature_series(idx_tst, feat=feat)
            axs[idx_tst].plot(x, y, marker='o', markersize=0.5, lw=0.125, label='Observed values')
            axs[idx_tst].xaxis.set_major_formatter(mdates.DateFormatter(rec.T_FMT))
            axs[idx_tst].set_title(f'Run-to-failure test {idx_tst + 1}')

            onset = p.degradation_onset_prev_(y, prev_tuned=feat)
            if onset != -1:
                axs[idx_tst].axvline(x=x[onset], c='r', label='Detected degradation onset', lw=0.5)

        plt.legend()
        title = f'{feat_disp} across all tests'
        plt.suptitle(title)
        ic(title)
        plt.savefig(f'plot/{title}', dpi=300)
        # plt.show()

