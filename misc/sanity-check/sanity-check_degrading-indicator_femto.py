"""
Sanity check on refined final output of FEMTO degradation times
"""

from scipy.stats import kurtosis as kurt, skew, skewtest
from unidip import UniDip

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
    t.normality_visual(vals, title=f'FEMTO failed bearing {idx_tst+1}, {rec.FEAT_DISP_NMS[feat]}', save=False)


def test_all():
    for idx_tst in range(rec.NUM_BRG_TST):
        for feat in indicators_degrading_new:
            onset = onsets[str(idx_tst)]
            vals = rec.get_feature_series(idx_tst, feat=feat)
            vals_h = vals[:onset]
            # ic(t.normality_shapiro_wilk(vals_healthy), t.normality_DAgostino_K2(vals_healthy))
            # ic(t.normality_AndersonDarling(vals_healthy))
            # t.normal_enough(vals)

            if not t.trend_normal_enough(vals_h, z=5):
                ic(idx_tst, feat, onset)
                ic(len(UniDip(vals_h, alpha=0.05).run()))
                ic(skew(vals_h))
                # ic(skewtest(vals_h))
                ic('')

            # if not t.normal_enough(vals):
            #     ic(idx_tst, feat, onset)
            #     ic(UniDip(vals, alpha=0.20).run())
            #     ic(skew(vals))
            #     # ic(skewtest([100, 100, 100, 100, 100, 100, 100, 101]))
            #     ic()
                # ic(kurt(vals_h), skew(vals_h))
                # ic(kurt(vals), skew(vals))

            # title = f'FEMTO failed bearing {idx_tst + 1}, {rec.FEAT_DISP_NMS[feat]}, all data'
            # t.normality_visual(vals, title=title, save=True)
            # exit(1)


if __name__ == '__main__':
    rec = VibRecordFemto()
    indicators_degrading_new = config('femto.degrading_indicators_new')
    onsets = config('femto.degrading_onsets_new_subset')
    ic(indicators_degrading_new, onsets)
    t = VibTransfer()

    # test_single(0, 'range_time')
    test_all()




