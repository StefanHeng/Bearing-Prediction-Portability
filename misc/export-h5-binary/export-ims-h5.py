"""
Actually export the IMS dataset into HDF5 format
"""

import numpy as np

import matplotlib.pyplot as plt

from icecream import ic

from vib_export_ims import VibExportIms

if __name__ == '__main__':
    exp = VibExportIms()
    # arr = exp.get_test_feature_series(0)

    # x = np.arange(arr.shape[1])
    # plt.figure(figsize=(16, 9), constrained_layout=True)
    # for idx, row in enumerate(arr):
    #     ic(row)
    #     plt.plot(x, row, label=f'row{idx}')
    # plt.legend(loc=0)
    # plt.show()

    # arr_rms = exp.get_feature_series(0, 0, ch=0)
    # arr_rms = exp.get_feature_series(1, 2, ch=1)
    # ic(arr_rms, arr_rms.shape)

    exp.export()

