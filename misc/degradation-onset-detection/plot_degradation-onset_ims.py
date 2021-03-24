import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import os
from icecream import ic

from vib_record_ims import VibRecordIms
from vib_predict import VibPredict

os.chdir('../..')


if __name__ == '__main__':
    rec = VibRecordIms()
    p = VibPredict()

    idx_tst = 3
    feat = 'kurtosis'

    x = rec.get_time_axis(idx_tst=idx_tst)

    fig, axs = plt.subplots(rec.NUM_BRG, figsize=(16, 12), constrained_layout=True)
    for idx_brg in range(rec.NUM_BRG):
        y = rec.get_feature_series(idx_tst, idx_brg, feat=feat)
        axs[idx_brg].plot(x, y, marker='o', markersize=0.5, linewidth=0.125)
        axs[idx_brg].xaxis.set_major_formatter(mdates.DateFormatter(rec.T_FMT))

        onset = p.degradation_onset_prev(y)
        ic(idx_brg, onset)
        if onset != -1:
            axs[idx_brg].axvline(x=x[onset], color='r', label='Detected degradation onset', linewidth=0.5)
        axs[idx_brg].set_title(f'Bearing {idx_brg+1}')
        axs[idx_brg].set_ylim([0, 10])

    plt.legend(loc=0)
    title = f'Degradation onset detected using [{feat}] on test {idx_tst+1}_IMS'
    plt.suptitle(title)
    plt.savefig(f'plot/{title}', dpi=300)
    plt.show()
