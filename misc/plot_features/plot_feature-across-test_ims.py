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
    rec = VibRecordIms()
    p = VibPredict()

    n_plots = sum([len(rec.BRGS_FLD[idx_tst]) for idx_tst in range(rec.NUM_TST)])
    ic(n_plots)
    for feat, feat_disp in rec.FEAT_DISP_NMS.items():
        i = 0
        fig, axs = plt.subplots(n_plots, figsize=(16, 12), constrained_layout=True)
        for idx_tst in range(rec.NUM_TST):
            idxs_brg = rec.BRGS_FLD[idx_tst]
            for m, idx_brg in enumerate(idxs_brg):
                x = rec.get_time_axis(idx_tst=idx_tst)
                y = rec.get_feature_series(idx_tst, idx_brg, feat=feat)
                axs[i].plot(x, y, marker='o', markersize=0.5, linewidth=0.125)

                axs[i].xaxis.set_major_formatter(mdates.DateFormatter(rec.T_FMT))
                axs[i].set_title(f'Test {idx_tst+1}, failed bearing {idx_brg+1}')
                i += 1

        title = f'{feat_disp} across all tests_IMS'
        plt.suptitle(title)
        ic(title)
        plt.savefig(f'plot/{title}', dpi=300)
