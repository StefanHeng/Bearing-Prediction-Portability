import numpy as np

import matplotlib.pyplot as plt

import os

from icecream import ic

from vib_record_ims import VibRecordIms

os.chdir('../..')


if __name__ == "__main__":
    rec = VibRecordIms()

    for idx_tst in range(rec.NUM_TST):
        x = rec.get_time_axis(idx_tst=idx_tst)
        for idx_brg in rec.BRGS_FLD[idx_tst]:
            for feat, feat_disp_nm in rec.FEAT_DISP_NMS.items():
                title = f'Test {idx_tst + 1}, failed bearing {idx_brg+1} {feat_disp_nm}'
                ic(idx_tst, title)

                plt.figure(figsize=(16, 9), constrained_layout=True)
                plt.plot(x, rec.get_feature_series(idx_tst, idx_brg, feat=feat), 'o',
                         label=f'Bearing {idx_brg + 1}', markersize=0.5)
                plt.suptitle(title)
                # plt.legend(loc=0)
                plt.savefig(f'plot/{title}.png', dpi=300)
                plt.close()
                # plt.show()
