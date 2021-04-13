"""
Sanity check on the relative ratio of refined final output of FEMTO degradation times,
to check degradation onset detection algorithm performance
"""

import numpy as np

import matplotlib.pyplot as plt
import os
from icecream import ic

from util import config
from vib_record_ims import VibRecordIms
from vib_transfer import VibTransfer
from vib_predict import VibPredict


os.chdir('../..')


def test_all():
    for idx_tst, onset in onsets.items():
        onset = list(onset.values())[0]
        idx_tst = max(0, int(idx_tst)-1)
        sz = rec.NUMS_MSR[idx_tst]
        if not t.evaluate_degradation_onset(sz, onset):
            ret = t.evaluate_degradation_onset(sz, onset)
            ic(idx_tst, sz, onset, ret, onset / sz)


if __name__ == '__main__':
    rec = VibRecordIms()
    bearings_failed = config('ims.bearings_failed')
    indicators_degrading = config('ims.degrading_indicators')
    onsets = config('ims.degrading_onsets_detected')
    ic(bearings_failed, onsets)
    t = VibTransfer()

    test_all()
