import numpy as np
import pandas as pd

import os
import h5py
import json

from icecream import ic

from dev_link import *
from util import *


class VibRecordIms:
    """ Reads vibration features extracted over time from `h5` file exported by `VibExp`
    """

    BRGS_FLD = {  # Bearing indices with observed failure for each test
        0: [2, 3],
        1: [2, 3],  # Essentially the same test as 0
        2: [0],
        3: [2]
    }

    T_FMT = '%d %H:%M'

    def __init__(self, path=H5FEAT_PATH_IMS):
        self.record = h5py.File(path, 'r')
        self.FEAT_STOR_IDXS = json.loads(self.record.attrs['feat_stor_idxs'])
        self.FEAT_NMS = list(self.FEAT_STOR_IDXS.keys())
        # config = open('config.json', 'r')
        self.FEAT_DISP_NMS = config('feature_display_names')
        # List of bearing training test; Use as indices into h5 file
        self.TST_NMS = json.loads(self.record.attrs['tst_nms'])
        self.NUMS_MSR = json.loads(self.record.attrs['nums_msr'])  # Number of measurement for each bearing by index
        # ic(self.FEAT_STOR_IDXS, self.FEAT_DISP_NMS, self.TST_NMS, self.NUMS_MSR)
        ic(self.NUMS_MSR)
        self.NUM_TST = len(self.TST_NMS)
        self.NUM_BRG = config('ims.num_bearings')
        self.NUM_FEAT = len(self.FEAT_STOR_IDXS)

    def get_feature_series(self, idx_tst, idx_brg, feat='rms_time'):
        """
        :param idx_tst: Index for a test in [0, 3], specified by `VibRecordIms`
        :param idx_brg: The bearing test specified by index
        :param feat: Feature/Property
        :return: Array of the feature in question across the entire bearing test, in sequential time
        """
        if feat not in self.FEAT_NMS:
            raise Exception(f'feat={feat} not found: feat has to be in {self.FEAT_NMS}')
        tst_nm = f'test{self._get_h5_test_idx(idx_tst)+1}'
        if idx_tst in [0, 1]:  # The 1st test in IMS
            tst_nm += f'/ch{idx_tst+1}'
        return self.record[f'{tst_nm}/bearing{idx_brg+1}'][self.FEAT_STOR_IDXS[feat]]

    @staticmethod
    def _get_h5_test_idx(idx_tst):
        return max(idx_tst-1, 0)

    def get_time_axis(self, strt=0, end=-1, inc=10, idx_tst=None):
        """
        :param strt: Index corresponding to start time by multiplying inc
        :param end: Index corresponding to end time by multiplying inc
        :param inc: Difference between 2 consecutive time stamps, in minutes
        :param idx_tst: Index of test, if specified, the time axis range is inferred with the full test duration
        :return: Inclusive-exclusive pandas time stamps

        .. note:: Time stamps are with respect to epoch time,
        since plotting libraries don't support plotting `timedelta`s
        """
        if idx_tst is not None:
            x = np.arange(self.NUMS_MSR[self._get_h5_test_idx(idx_tst)])
        else:
            x = np.arange(strt, end)
        return pd.to_datetime(pd.Series(x * inc), unit='m')
