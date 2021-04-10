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

    indicators_degrading = config('ims.degrading_indicators')
    params = config('ims.prev_hyperparameters')
    onsets = dict()

    ic(indicators_degrading, params)
    for idx_tst in range(rec.NUM_TST):
        idxs_brg = rec.BRGS_FLD[idx_tst]
        l = len(indicators_degrading)
        fig, axs = plt.subplots(l * len(idxs_brg), figsize=(16, 12), constrained_layout=True)
        for m, idx_brg in enumerate(idxs_brg):
            for idx, feat in enumerate(indicators_degrading):
                idx += m * l
                feat_disp = rec.FEAT_DISP_NMS[feat]
                x = rec.get_time_axis(idx_tst=idx_tst)
                y = rec.get_feature_series(idx_tst, idx_brg, feat=feat)
                axs[idx].plot(x, y, marker='o', markersize=0.5, linewidth=0.125)

                onset = p.degradation_onset_prev_(y, prev_tuned=feat)
                if onset != -1:
                    axs[idx].axvline(x=x[onset], color='r', label='Degradation onset detected', linewidth=0.5)

                ic(idx_tst, idx_brg, feat, onset)
                if idx_tst not in onsets:
                    onsets[idx_tst] = dict()
                if idx_brg not in onsets[idx_tst]:
                    onsets[idx_tst][idx_brg] = dict()
                onsets[idx_tst][idx_brg][feat] = onset

                axs[idx].xaxis.set_major_formatter(mdates.DateFormatter(rec.T_FMT))
                axs[idx].set_title(f'{feat_disp}, failed bearing {idx_brg+1}')
        plt.legend()

        title = f'Degradation onset previous method with tuned hyperparameters, test {idx_tst + 1}'
        plt.suptitle(title)
        ic(title)
        plt.savefig(f'plot/{title}', dpi=300)
        # plt.show()

    ic(onsets)

