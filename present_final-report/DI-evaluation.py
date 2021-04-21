"""
Sanity check on the trend in the healthy stage, based on refined final output of FEMTO degradation times,
to confirm degrading indicator selection
"""

from scipy.stats import kurtosis as kurt, skew, skewtest
from unidip import UniDip

import os
from icecream import ic

from util import *
from vib_record_femto import VibRecordFemto
from vib_transfer import VibTransfer

os.chdir('..')


if __name__ == '__main__':
    rec = VibRecordFemto()
    indicators_degrading_new = config('femto.degrading_indicators_new')
    onsets = config('femto.degrading_onsets_new_subset')
    ic(indicators_degrading_new, onsets)
    t = VibTransfer()

    idx_tst = 0
    feat = 'peak_freq'
    onset = onsets[str(idx_tst)]
    vals = rec.get_feature_series(idx_tst, feat=feat)[:onset]
    feat_d = rec.FEAT_DISP_NMS[feat]
    tt = 'Normality check of healthy stage values on the FEMTO dataset'
    t.normality_visual(vals, title=f'Run-to-failure test {rec.tst_nm(idx_tst)}: {feat_d}',
                       path=f'present_final-report/{tt}', xlab=feat_d, save=True)
