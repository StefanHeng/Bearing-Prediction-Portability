import numpy as np

import os
from icecream import ic

from vib_record_femto import VibRecordFemto
from vib_record_ims import VibRecordIms
from vib_transfer import VibTransfer


os.chdir('../..')


if __name__ == "__main__":
    rec = VibRecordFemto()
    t = VibTransfer()
    # ic(t.degrading_detection_fitness(rec.get_feature_series(0, feat='rms_time'), plot='RMS in time, bearing 0'))
    # ic(t.degrading_detection_fitness(rec.get_feature_series(0, feat='mean_freq'), plot='Mean in freq, bearing 0'))

    # C = np.array([1, 2, 1, 2, 1]) / 7
    # C = np.array([3, 3, 1]) / 7
    # ic(t.degrad_detect_score(rec.get_feature_series(0, feat='rms_time'), C=C, plot='RMS in time'))
    # ic(t.degrad_detect_score(rec.get_feature_series(0, feat='mean_freq'), C=C, plot='Mean in freq'))

    # ic()
    for idx, feat in enumerate(rec.FEAT_NMS):
        metrics = []
        for idx_brg in range(rec.NUM_BRG_TST):
            for acc in ['h', 'v']:
                nm = f'{rec.FEAT_DISP_NMS[feat]} on test {idx_brg + 1} {acc}'
                # ic(nm)
                nm = ''
                m = t.degrading_detection_fitness(rec.get_feature_series(idx_brg, feat=feat, acc=acc), plot=nm)
                metrics.append(m)
        m = np.array(metrics).mean()
        ic(feat, m)
    # ic()

    rec_ims = VibRecordIms()
