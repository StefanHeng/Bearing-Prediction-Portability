"""
Sanity check on refined final output of FEMTO degradation times
"""

import os
from icecream import ic

from util import *
from vib_record_femto import VibRecordFemto
from vib_transfer import VibTransfer

os.chdir('../..')


def test_single(idx_tst, feat):
    onset = onsets[str(idx_tst)]
    vals = rec.get_feature_series(idx_tst, feat=feat)[:onset]

    ic(t.normality_ShapiroWilk(vals), t.normality_DAgostinoK2(vals), t.normality_AndersonDarling(vals))
    t.normality_visual(vals, save='FEMTO degrading')


def test_all():
    for idx_tst in range(rec.NUM_BRG_TST):
        for feat in indicators_degrading_new:
            onset = onsets[str(idx_tst)]
            vals_healthy = rec.get_feature_series(idx_tst, feat=feat)[:onset]
            ic(idx_tst, feat, onset)
            # ic(t.normality_shapiro_wilk(vals_healthy), t.normality_DAgostino_K2(vals_healthy))
            ic(t.normality_AndersonDarling(vals_healthy))
            exit(1)


if __name__ == '__main__':
    rec = VibRecordFemto()
    indicators_degrading_new = config('femto.degrading_indicators_new')
    onsets = config('femto.degrading_onsets_new_subset')
    ic(indicators_degrading_new, onsets)
    t = VibTransfer()

    test_single(0, 'range_time')
    test_all()




