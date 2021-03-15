import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from icecream import ic

from vib_record import VibRecord

if __name__ == '__main__':
    rec = VibRecord()
    idx_brg = 5  # On 1st test, look at horizontal vibration
    acc = 'hori'
    feat = 'mean_freq'
    idx_feat = rec.FEAT_STOR_IDXS[feat]

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    x = rec.get_time_axis(0, rec.NUMS_MSR[idx_brg])

    vals = rec.get_feature_series(idx_brg, feat=feat, acc=acc)
    plt.scatter(x, vals, s=0.25)
    ax.set_xlabel('time (HH:MM:SS)')
    feat_disp_nm = rec.FEAT_DISP_NMS[str(idx_feat)]
    ax.set_ylabel(feat_disp_nm)
    title = f'{feat_disp_nm} on {acc} bearing {idx_brg+1}'
    fig.suptitle(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.savefig(f'present/Feature selection against time, {title}.png', dpi=300)
    plt.show()

