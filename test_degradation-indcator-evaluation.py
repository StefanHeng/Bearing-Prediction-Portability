import numpy as np

from vib_record import VibRecord
from vib_transfer import VibTransfer


from icecream import ic


if __name__ == "__main__":
    rec = VibRecord()
    t = VibTransfer()
    for feat in rec.FEAT_NMS:
        metrics = []
        for i in range(rec.NUM_BRG_TST):
            metrics.append(t.degrad_diff(rec.get_feature_series(i, feat=feat)))
        m = np.array(metrics).mean()
        ic(feat, m)

