import numpy as np
import pandas as pd

from os import path
import h5py
import json

from icecream import ic

from dev_link import *
from vib_extract import VibExtractFemto


class VibExportFemto:
    """ Handles reading FEMTO vibration dataset in raw and exporting to h5py data record,
    with properties specified in `VibExtr`.

    Each roll bearing's measurement recorded at a time will be 1-dimensional for splitting
    horizontal and vertical measurements
    """

    FLDR_NMS = [  # Maps index of a bearing on training data to the folder name on storage
        'Bearing1_1',
        'Bearing1_2',
        'Bearing2_1',
        'Bearing2_2',
        'Bearing3_1',
        'Bearing3_2'
    ]
    ENC_FEAT_STOR = dict(  # Dictionary on storage encoding of properties
        rms_time=0,
        range_time=1,
        kurtosis=2,
        skewness=3,
        rms_freq=4,
        mean_freq=5,
        peak_freq=6
    )

    CD_H = ['h', 'hori']  # Code for horizontal acceleration

    def __init__(self):
        self.NUMS_MESR = self._get_num_acc_files()  # Number of acceleration measurements, the final count for each

    def _get_num_acc_files(self):
        nums = []
        for fd_nm in self.FLDR_NMS:
            num = 0
            while path.exists(f'{DATA_PATH}{fd_nm}/acc_{num+1:05}.csv'):
                num += 1
            nums.append(num)
        return nums

    def get_single_measurement(self, n, idx_brg):
        fl_nm = f'{DATA_PATH}{self.FLDR_NMS[idx_brg]}/acc_{n+1:05}.csv'
        return pd.read_csv(fl_nm, header=None)

    def get_vib_values(self, n, idx_brg, acc='hori'):
        """
        :param n: 0-indexed measurement number, by sequential time
        :param idx_brg: Encoding for the 6 roll bearing tests, takes on [0, 5]
        :param acc: Takes value in {'h', 'hori', 'v', 'vert'} for horizontal or vertical acceleration
        """
        fl_nm = f'{DATA_PATH}{self.FLDR_NMS[idx_brg]}/acc_{n+1:05}.csv'
        df = pd.read_csv(fl_nm, header=None)
        return (df[4] if acc in self.CD_H else df[5]).to_numpy()

    def get_feature_series(self, idx_brg, func_extr, acc='hori'):
        """
        :param idx_brg: The bearing test specified by index
        :param func_extr: The function in `VibExtr` corresponding to a property to extract on a single measurement
        :param acc: Specified horizontal or vertical acceleration
        :return: Array of the feature in question across the entire bearing test, in sequential time
        """
        idxs = np.arange(self.NUMS_MESR[idx_brg])
        return np.vectorize(lambda i: func_extr(self.get_vib_values(i, idx_brg, acc)))(idxs)

    def export(self, fl_nm='femto_features_train', num_spl=N_SPL, spl_rt=SPL_RT):
        """ One group for each bearing test, with all the properties by horizontal/vertical
        """
        extr = VibExtractFemto(num_spl, spl_rt)
        fl_nm = f'data/{fl_nm}.hdf5'
        open(fl_nm, 'a').close()  # Create file in OS
        fl = h5py.File(fl_nm, 'w')
        fl.attrs['feat_stor_idxs'] = json.dumps(self.ENC_FEAT_STOR)
        fl.attrs['brg_nms'] = json.dumps(self.FLDR_NMS)
        fl.attrs['nums_msr'] = json.dumps(self.NUMS_MESR)
        fl.attrs['feat_disp_nms'] = json.dumps({idx: nm for idx, nm in enumerate(extr.D_PROP_FUNC)})
        print(f'Metadata attributes created: {list(fl.attrs.keys())}')
        for idx_brg, test_nm in enumerate(self.FLDR_NMS):
            group = fl.create_group(test_nm)
            for acc in ['hori', 'vert']:
                arr_extr = np.stack([
                    self.get_feature_series(idx_brg, func, acc) for k, func in extr.D_PROP_FUNC.items()
                ])
                group.create_dataset(acc, data=arr_extr)
        print(f'Features extracted: {[nm for nm in fl]}')
