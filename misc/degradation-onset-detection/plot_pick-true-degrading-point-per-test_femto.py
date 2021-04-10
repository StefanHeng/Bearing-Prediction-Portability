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

    # indicators_degrading = config('femto.degrading_indicators')
    truth = config('femto.onset_truth')

    ic(truth)
    num_feat = config('femto.num_features')
    for idx_brg in range(rec.NUM_BRG_TST):
        fig, axs = plt.subplots(num_feat, figsize=(16, 12), constrained_layout=True)
        for idx, (feat, feat_disp) in enumerate(rec.FEAT_DISP_NMS.items()):
            # ic(idx, (feat, feat_disp))
            x = rec.get_time_axis(idx_brg=idx_brg)
            y = rec.get_feature_series(idx_brg, feat=feat)
            axs[idx].plot(x, y, marker='o', markersize=0.5, lw=0.125)

            onset = p.degradation_onset_prev_(y)
            if onset != -1:
                axs[idx].axvline(x=x[onset], color='r', label='Degradation detected, previous method',
                                 lw=0.5)
            axs[idx].axvline(x=x[truth[idx_brg]], color='g', label='Degradation onset manual ground truth',
                             lw=0.5)

            axs[idx].xaxis.set_major_formatter(mdates.DateFormatter(rec.T_FMT))
            axs[idx].set_title(feat_disp)
        plt.legend()

        title = f'Degradation onset manual ground truth, run-to-failure test {idx_brg + 1}'
        plt.suptitle(title)
        ic(title)
        plt.savefig(f'plot/{title}', dpi=300)
        # plt.show()

        # break
