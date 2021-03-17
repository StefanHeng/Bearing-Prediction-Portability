import numpy as np

from vib_record import VibRecord
from vib_transfer import VibTransfer


from icecream import ic


if __name__ == "__main__":
    rec = VibRecord()
    t = VibTransfer()
    ic(t.degrad_detect_score(rec.get_feature_series(0, feat='rms_time'), plot='RMS in time, bearing 0'))
    ic(t.degrad_detect_score(rec.get_feature_series(0, feat='mean_freq'), plot='Mean in freq, bearing 0'))

    # C = np.array([1, 2, 1, 2, 1]) / 7
    # C = np.array([3, 3, 1]) / 7
    # ic(t.degrad_detect_score(rec.get_feature_series(0, feat='rms_time'), C=C, plot='RMS in time'))
    # ic(t.degrad_detect_score(rec.get_feature_series(0, feat='mean_freq'), C=C, plot='Mean in freq'))

    # ic()
    for idx, feat in enumerate(rec.FEAT_NMS):
        metrics = []
        for i in range(rec.NUM_BRG_TST):
            for acc in ['h', 'v']:
                nm = f'{rec.FEAT_DISP_NMS[str(idx)]} on test {i+1} {acc}'
                # ic(nm)
                nm = ''
                m = t.degrad_detect_score(rec.get_feature_series(i, feat=feat, acc=acc), plot=nm)
                metrics.append(m)
        m = np.array(metrics).mean()
        ic(feat, m)
    # ic()
