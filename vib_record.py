import numpy as np
import pandas as pd

import h5py
import json

from icecream import ic

from dev_link import *


class VibRecord:
    """ Reads vibration features extracted over time from `h5` file exported by `VibExp`
    """
    def __init__(self, path=FEAT_PATH):
        self.record = h5py.File(path, 'r')
        self.FEAT_STOR_IDXS = json.loads(self.record.attrs['feat_stor_idxs'])
        self.FEAT_DISP_NMS = json.loads(self.record.attrs['feat_disp_nms'])
        # List of bearing training test; Use as indices into h5 file
        self.BRG_NMS = json.loads(self.record.attrs['brg_nms'])
        self.NUMS_MSR = json.loads(self.record.attrs['nums_msr'])  # Number of measurement for each bearing by index
        # ic(self.FEAT_STOR_IDXS, self.FEAT_DISP_NMS, self.BRG_NMS, self.NUMS_MSR)

    def get_feature_series(self, idx_brg, feat='rms_time', acc='hori'):
        """
        :param feat: The feature/property in question
        :param idx_brg: The bearing test specified by index
        :param acc: Specified horizontal or vertical acceleration
        :return: Array of the feature in question across the entire bearing test, in sequential time
        """
        return self.record[f'{self.BRG_NMS[idx_brg]}/{acc}'][self.FEAT_STOR_IDXS[feat]]

    @staticmethod
    def get_time_axis(strt, end, inc=10):
        """
        :param strt: Index corresponding to start time by multiplying inc
        :param end: Index corresponding to end time by multiplying inc
        :param inc: Difference between 2 consecutive time stamps, in minutes
        :return: Inclusive-exclusive pandas time stamps

        .. note:: Time stamps are with respect to epoch time, since plotting libraries don't support plotting `timedelta`s
        """
        x = np.arange(strt, end) * inc
        return pd.to_datetime(pd.Series(x), unit='m')

