import numpy as np

import os

from vib_record import VibRecord
from vib_record_ims import VibRecordIms
from vib_transfer import VibTransfer


from icecream import ic


os.chdir('../..')


if __name__ == "__main__":
    rec = VibRecordIms()
    t = VibTransfer()

    # ic()
    for feat in rec.FEAT_DISP_NMS:
        metrics = []
        weights = []
        num_elm = 0
        for idx_tst in range(rec.NUM_TST):
            for idx_brg in rec.BRGS_FLD[idx_tst]:
                m = t.degrad_detect_score(rec.get_feature_series(idx_tst, idx_brg, feat=feat))
                # ic(feat, idx_tst, idx_brg, m)
                if idx_tst in [0, 1]:  # The values are highly correlated
                    weights.append(0.5)  # Split the weight
                    num_elm += 0.5
                else:
                    weights.append(1)
                    num_elm += 1
                metrics.append(m)
                # ic(idx_tst, metrics, weights, num_elm)
        metrics = np.array(metrics)
        weights = np.array(weights)
        m = np.dot(metrics, weights) / num_elm  # test1 and test1 considered single test
        ic(feat, m)
    # ic()


