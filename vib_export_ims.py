import numpy as np
import pandas as pd

import h5py
import json

from dev_link import *
from ims_attrs import *

from icecream import ic


class VibExportIms:
    """ Handles reading IMS dataset feature extraction output in CSV (by previous project)
    and exporting to h5py data record,
    with the following properties: RMS_time, Range_time, Kurtosis, Skewness, Range_freq, Peak_freq, Mean_freq, Peak_amp

    .. note:: The 1st test has 2 dimension for each of the 4 bearings
    .. note:: The 4th test is defined test 3

    .. note:: The 1st row in the feature domain intentionally removed was included for export

    Split into 4 partitions with names: [test1/ch1/, test1/ch2/, test2/, test3/]
    """
    FEAT_CSV_NMS = dict(
        rms_time='RMS',
        range_time='p2p',
        kurtosis='kurtosis',
        skewness='skew',
        rms_freq='rmsFeq',
        mean_freq='meanFeq',
        peak_freq='peakFeq',
        rot_amp='peakAmp'
    )

    ENC_FEAT_STOR = dict(  # Dictionary on storage encoding of properties
        rms_time=0,
        range_time=1,
        kurtosis=2,
        skewness=3,
        rms_freq=4,
        mean_freq=5,
        peak_freq=6,
        rot_amp=7
    )

    def get_test_feature_series(self, idx_tst, feat='rms_time'):
        """ Feature across all channels available """
        fl_nm = f'{FEAT_PATH_IMS}test{idx_tst+1}_{self.FEAT_CSV_NMS[feat]}.csv'
        df = pd.read_csv(fl_nm, header=None)
        return df.to_numpy().T

    def get_feature_series(self, idx_tst, idx_brg, feat='rms_time', ch=None):
        """
        :param idx_tst: The test number index in [0, 2] for 3 tests
        :param idx_brg: The bearing index in [0, 3] for 4 bearings
        :param feat: Feature/Property
        :param ch: Which channel of bearing vibration, integer in [0, 1], only an option for test 1
        """
        idx = idx_brg if idx_tst != 0 else (idx_brg * 2 + ch)
        fl_nm = f'{FEAT_PATH_IMS}test{idx_tst+1}_{self.FEAT_CSV_NMS[feat]}.csv'
        df = pd.read_csv(fl_nm, header=None)
        return df[idx].to_numpy()

    def export(self, fl_nm='ims_features'):
        fl_nm = f'data/{fl_nm}.hdf5'
        open(fl_nm, 'a').close()  # Create file in OS
        fl = h5py.File(fl_nm, 'w')
        fl.attrs['feat_stor_idxs'] = json.dumps(self.ENC_FEAT_STOR)
        fl.attrs['nums_msr'] = json.dumps(NUMS_MESR)
        tst_nms = []
        for (idx_tst, tst_nm) in [(0, '/ch1'), (1, '/ch2'), (2, ''), (3, '')]:
            i_tst = max(idx_tst - 1, 0)  # To be compatible with `get_feature_series`
            tst_nm = f'test{i_tst+1}{tst_nm}'
            tst_nms.append(tst_nm)
            group = fl.create_group(tst_nm)
            idxs_brg = list(range(NUM_BRG))
            # if idx_tst == 0:
            #     idxs_brg = [i * 2 for i in idxs_brg]
            # elif idx_tst == 1:
            #     idxs_brg = [i * 2 + 1 for i in idxs_brg]
            for idx_brg in idxs_brg:
                dset = np.stack([
                    self.get_feature_series(i_tst, idx_brg, feat=f, ch=idx_tst) for f in self.FEAT_CSV_NMS
                ])
                group.create_dataset(f'bearing{idx_brg+1}', data=dset)
        fl.attrs['tst_nms'] = json.dumps(tst_nms)  # 4 tests
        print(f'Metadata attributes created: {list(fl.attrs.keys())}')
        print(f'Features extracted: {[nm for nm in fl]}')
