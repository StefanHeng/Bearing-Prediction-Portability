"""
Sanity check on the trend in the healthy stage, based on refined final output of FEMTO degradation times,
to confirm degrading indicator selection
"""

from scipy.stats import kurtosis as kurt, skew

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

    # ic(t.normality_ShapiroWilk(vals), t.normality_DAgostinoK2(vals), t.normality_AndersonDarling(vals))
    t.normality_visual(vals, title=f'IMS test {idx_tst+1}, failed bearing {idx_brg+1}, {rec.FEAT_DISP_NMS[feat]}')


def test_all():
    for idx_tst, onsets_brg in onsets.items():
        for idx_brg, onset in onsets_brg.items():
            onset = onsets[idx_tst][idx_brg]
            i_tst, idx_brg = int(idx_tst), int(idx_brg)
            for feat in indicators_degrading:
                vals_h = rec.get_feature_series(i_tst, idx_brg, feat=feat)[:onset]
                if not t.trend_normal_enough(vals_h, z=5):
                    ic(i_tst, idx_brg, feat, onset)

                feat_disp = rec.FEAT_DISP_NMS[feat]
                title = f'IMS visual normality check test {i_tst+1}, failed bearing {idx_brg+1}, {feat_disp}'
                t.normality_visual(vals_h, title=title, save=True)
                # ic(t.normality_shapiro_wilk(vals_healthy), t.normality_DAgostino_K2(vals_healthy))
                # ic(t.normality_AndersonDarling(vals_healthy))


if __name__ == '__main__':
    rec = VibRecordIms()
    bearings_failed = config('ims.bearings_failed')
    indicators_degrading = config('ims.degrading_indicators')
    onsets = config('ims.degrading_onsets_detected')
    ic(bearings_failed, indicators_degrading, onsets)
    t = VibTransfer()

    # test_single(0, 2, 'range_time')
    test_all()




