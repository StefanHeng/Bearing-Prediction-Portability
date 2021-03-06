import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from icecream import ic

from vib_record import VibRecord

if __name__ == '__main__':
    rec = VibRecord()
    idx_brg = 0  # On 1st test, look at horizontal vibration
    acc = 'hori'

    # for idx_brg in range(6):
    #     for acc in ['hori', 'vert']:
    fig, axs = plt.subplots(len(rec.FEAT_STOR_IDXS), figsize=(16, 13), constrained_layout=True)
    x = rec.get_time_axis(0, rec.NUMS_MSR[idx_brg])
    # ic(x)

    for feat, idx in rec.FEAT_STOR_IDXS.items():
        # ic(feat, idx)
        vals = rec.get_feature_series(idx_brg, feat=feat, acc=acc)
        axs[idx].scatter(x, vals, s=0.25)
        axs[idx].set_xlabel('time (<Day> <Hour>:<Min>)')
        axs[idx].set_ylabel(rec.FEAT_DISP_NMS[str(idx)])
        axs[idx].xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M"))
    plt.savefig(f'plot/Feature selection against time, {acc} bearing {idx_brg+1}.png', dpi=300)
    plt.show()

