import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

                fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
                plt.plot(x, rec.get_feature_series(idx_tst, idx_brg, feat=feat), marker='o',
                         label=f'Bearing {idx_brg + 1}', markersize=0.5, linewidth=0.125)
                # plt.plot(x, rec.get_feature_series(idx_tst, idx_brg, feat=feat),
                #          label=f'Bearing {idx_brg + 1}', linewidth=0.125)
                plt.suptitle(title)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M"))
                plt.savefig(f'plot/{title}.png', dpi=300)
                plt.close()
                # plt.show()
