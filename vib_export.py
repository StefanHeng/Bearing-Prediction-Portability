import pandas as pd

from glob import glob
from os import path

from icecream import ic

from dev_link import *


class VibExp:
    """ Handles reading FEMTO vibration dataset in raw and exporting to h5py data record,
    with properties specified in `VibExtr`.

    Each roll bearing's measurement recorded will be 1-dimensional.

    Each acceleration file has 0.1s of data every 10 min.

    References
    ----------
    .. [1] A. Saxena and K. Goebel (2008). "PHM08 Challenge Data Set", NASA Ames Prognostics Data Repository
            (http://ti.arc.nasa.gov/project/prognostic-data-repository), NASA Ames Research Center, Moffett Field, CA
    """
    SPL_RT = 25_600  # Sample rate
    NUM_BRG = 6  # Number of roll bearings in training data
    N_SPL = SPL_RT // 10  # 0.1 second of data

    FLDR_NMS = [  # Maps index of a bearing on training data to the folder name on storage
        'Bearing1_1',
        'Bearing1_2',
        'Bearing2_1',
        'Bearing2_2',
        'Bearing3_1',
        'Bearing3_2'
    ]

    CD_H = ['h', 'hori']  # Code for horizontal acceleration

    def __init__(self):
        self.NUMS_FL = self._get_num_acc_files()  # Number of acceleration files, the final count for each

    def _get_num_acc_files(self):
        nums = []
        for fd_nm in self.FLDR_NMS:
            num = 1
            while path.exists(f'{DATA_PATH}{fd_nm}/acc_{num:05}.csv'):
                num += 1
            nums.append(num)
        return nums

    def get_vib_values(self, n, idx_brg, acc='hori'):
        """
        :param n: 0-indexed measurement number, by sequential time
        :param idx_brg: Encoding for the 6 roll bearing tests, takes on [0, 5]
        :param acc: Takes value in {'h', 'hori', 'v', 'vert'} for horizontal or vertical acceleration
        """
        fl_nm = f'{DATA_PATH}{self.FLDR_NMS[idx_brg]}/acc_{n+1:05}.csv'
        df = pd.read_csv(fl_nm, header=None)
        return (df[4] if acc in self.CD_H else df[5]).to_numpy()

    # def get_feature_series(self, idx_brg, func_extr):
    #     """
    #     :param idx_brg: The bearing test to pick from
    #     :param func_extr: The function in `VibExtr` corresponding to a property to extract on a single measurement
    #     :return: Array of the feature in question, in sequential time
    #     """
