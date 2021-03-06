import h5py
import json

from icecream import ic

from dev_link import *


class VibRecord:
    """ Reads vibration features extracted over time from `h5` file exported by `VibExp`
    """
    def __init__(self, path=FEAT_PATH):
        self.record = h5py.File(path, 'r')
        self.feat_map = json.loads(self.record.attrs['feat_map'])
        # List of bearing training test; Use as indices into h5 file
        self.brg_nms = json.loads(self.record.attrs['brg_nms'])
        # ic(self.feat_map, self.brg_nms)

    def get_feature_series(self, idx_brg, prop='rms_time', acc='hori'):
        """
        :param prop: The feature/property in question
        :param idx_brg: The bearing test specified by index
        :param acc: Specified horizontal or vertical acceleration
        :return: Array of the feature in question across the entire bearing test, in sequential time
        """
        # ic(list(self.record.keys()))
        # ic(self.record[self.brg_nms[idx_brg]])
        # ic([nm for nm in self.record])
        return self.record[f'{self.brg_nms[idx_brg]}/{acc}'][self.feat_map[prop]]
