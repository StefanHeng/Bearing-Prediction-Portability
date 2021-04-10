"""
Use the previous degrading indicators selected, try to achieve a good degrading output on new dataset
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

    indicators_degrading = config('ims.degrading_indicators')
    truth = config('femto.onset_truth')  # As a rough guidance

    ic(indicators_degrading, truth)
    for idx_brg in range(rec.NUM_BRG_TST):
        fig, axs = plt.subplots(len(indicators_degrading), figsize=(16, 12), constrained_layout=True)
        for idx, feat in enumerate(indicators_degrading):
            feat_disp = rec.FEAT_DISP_NMS[feat]
            x = rec.get_time_axis(idx_brg=idx_brg)
            y = rec.get_feature_series(idx_brg, feat=feat)
            axs[idx].plot(x, y, marker='o', markersize=0.5, linewidth=0.125)

            onset = p.degradation_onset_prev_(y, sz_base=200, sz_window=15, z=3)
            if onset != -1:
                axs[idx].axvline(x=x[onset], color='r', label='Degradation onset detected',
                                 linewidth=0.5)
            axs[idx].axvline(x=x[truth[idx_brg]], color='g', label='Degradation onset ground truth',
                             linewidth=0.5)

            axs[idx].xaxis.set_major_formatter(mdates.DateFormatter(rec.T_FMT))
            axs[idx].set_title(feat_disp)
        plt.legend()

        title = f'FEMTO Degradation detection on previous indicators, tuned hyper-parameter, test bearing {idx_brg + 1}'
        plt.suptitle(title)
        ic(title)
        plt.savefig(f'plot/{title}', dpi=300)
        # plt.show()

        # break
