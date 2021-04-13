"""
Sanity check on the relative ratio of refined final output of FEMTO degradation times,
to check degradation onset detection algorithm performance
"""

import numpy as np

import matplotlib.pyplot as plt
import os
from icecream import ic

from util import config
from vib_record_femto import VibRecordFemto
from vib_transfer import VibTransfer
from vib_predict import VibPredict


os.chdir('../..')


def inspect_test2():
    """ Degradation onset output is not ideal, need to go back to tuning """
    p = VibPredict()
    idx_tst = 1
    y = rec.get_feature_series(idx_tst, feat='peak_freq')
    sz_window = 50
    x, y = p.sliding_means(y, sz_window=sz_window)

    plt.plot(x, y, marker='o', markersize=0.5, lw=0.125, label=f'Observed values, smoothed with window={sz_window}')
    plt.show()


def test_all():
    for idx_tst, onset in onsets.items():
        idx_tst = int(idx_tst)
        sz = rec.NUMS_MSR[idx_tst]
        if not t.evaluate_degradation_onset(sz, onset):
            ret = t.evaluate_degradation_onset(sz, onset)
            ic(idx_tst, sz, onset, ret, onset / sz)


if __name__ == '__main__':
    rec = VibRecordFemto()
    indicators_degrading_new = config('femto.degrading_indicators_new')
    onsets = config('femto.degrading_onsets_new_subset')
    ic(indicators_degrading_new, onsets)
    t = VibTransfer()

    # inspect_test2()
    test_all()
