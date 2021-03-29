import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import os
from icecream import ic

from util import *
from vib_record import VibRecord
# from vib_record_ims import VibRecordIms
from vib_predict import VibPredict

os.chdir('../..')

if __name__ == '__main__':
    rec = VibRecord()
    p = VibPredict()

    # feat = 'kurtosis'
    feat = 'skewness'
    # feat = 'rms_time'

    indicators_degrading = config('femto.degrading_indicators')
    truth = config('femto.onset_truth')

    ic(indicators_degrading, truth)
    for idx_brg in range(rec.NUM_BRG_TST):
        fig, axs = plt.subplots(len(indicators_degrading), figsize=(16, 12), constrained_layout=True)
        for idx, feat in enumerate(indicators_degrading):
            feat_disp = rec.FEAT_DISP_NMS[feat]
            x = rec.get_time_axis(idx_brg=idx_brg)
            y = rec.get_feature_series(idx_brg, feat=feat)
            axs[idx].plot(x, y, marker='o', markersize=0.5, linewidth=0.125)

            onset = p.degradation_onset_prev(y)
            if onset != -1:
                axs[idx].axvline(x=x[onset], color='r', label='Degradation onset detected, previous method',
                                 linewidth=0.5)
            axs[idx].axvline(x=x[truth[idx_brg]], color='g', label='Degradation onset manual ground truth',
                             linewidth=0.5)

            axs[idx].xaxis.set_major_formatter(mdates.DateFormatter(rec.T_FMT))
            axs[idx].set_title(feat_disp)
        plt.legend()

        title = f'Degradation onset manual ground truth, test bearing {idx_brg + 1}'
        plt.suptitle(title)
        ic(title)
        plt.savefig(f'plot/{title}', dpi=300)
        # plt.show()

        # break
