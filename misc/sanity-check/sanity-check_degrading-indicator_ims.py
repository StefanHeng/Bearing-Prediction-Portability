"""
Sanity check on refined final output of FEMTO degradation times
"""

import os
from icecream import ic

from util import *
# from vib_record_femto import VibRecordFemto
from vib_record_ims import VibRecordIms
from vib_transfer import VibTransfer

os.chdir('../..')


def test_single(idx_tst, idx_brg, feat):
    onset = onsets[str(idx_tst)][str(idx_brg)]
    vals = rec.get_feature_series(idx_tst, idx_brg, feat=feat)[:onset]

    ic(t.normality_ShapiroWilk(vals), t.normality_DAgostinoK2(vals), t.normality_AndersonDarling(vals))
    t.normality_visual(vals, title=f'Test {idx_tst+1}, failed bearing {idx_brg+1}, {rec.FEAT_DISP_NMS[feat]}')


def test_all():
    for idx_tst, idxs_brg in bearings_failed.items():
        idx_tst = max(0, int(idx_tst)-1)
        for idx_brg in idxs_brg:
            for feat in indicators_degrading:
                onset = onsets[str(idx_tst)][str(idx_brg)]
                vals_healthy = rec.get_feature_series(idx_tst, idx_brg, feat=feat)[:onset]
                ic(idx_tst, idx_brg, feat, onset)
                # ic(t.normality_shapiro_wilk(vals_healthy), t.normality_DAgostino_K2(vals_healthy))
                ic(t.normality_AndersonDarling(vals_healthy))
                exit(1)


if __name__ == '__main__':
    rec = VibRecordIms()
    bearings_failed = config('ims.bearings_failed')
    indicators_degrading = config('ims.degrading_indicators')
    onsets = config('ims.degrading_onsets_detected')
    ic(bearings_failed, indicators_degrading, onsets)
    t = VibTransfer()

    test_single(0, 2, 'range_time')
    test_all()




