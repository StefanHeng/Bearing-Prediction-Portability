"""
Sanity check by plot on the features selected for the IMS dataset
by VibTransfer fitness metric
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import os

from icecream import ic

from vib_record_ims import VibRecordIms
from vib_transfer import VibTransfer

os.chdir('../..')

if __name__ == "__main__":
    rec = VibRecordIms()
    t = VibTransfer()

    data_feat_selected = rec.get_feature_series(1, 3, feat='range_time')
    data_feat_okay = rec.get_feature_series(0, 3, feat='mean_freq')
    data_feat_not_fit = rec.get_feature_series(0, 2, feat='rms_freq')

    ic(t.degrading_detection_fitness(data_feat_selected, plot='Range in time, a good degradation onset indicator'))
    ic(t.degrading_detection_fitness(data_feat_okay, plot="Mean in frequency, an okay degradation onset indicator "
                                                          "(tho wasn't selected)"))
    ic(t.degrading_detection_fitness(data_feat_not_fit, plot='RMS in frequency, not fit for degradation onset'))
