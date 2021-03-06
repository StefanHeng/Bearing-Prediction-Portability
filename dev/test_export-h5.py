import numpy as np

import h5py
import json
import os

from icecream import ic

from vib_export import VibExport
from vib_extract import VibExtract

if __name__ == '__main__':
    exp = VibExport()
    extr = VibExtract()
    n = 0
    idx_brg = 0
    # series = exp.get_feature_series(idx_brg, extr.rms_time)
    # ic(series)

    fl_nm = 'test-lang.hdf5'
    open(fl_nm, 'w').close()
    f = h5py.File(fl_nm, 'w')
    enc_prop = dict(  # Dictionary on storage encoding of properties
        rms_time=0,
        range_time=1,
        kurtosis=2,
        skewness=3,
        rms_freq=4,
        mean_freq=5,
        peak_freq=6
    )
    f.attrs['property_map'] = json.dumps(enc_prop)
    # ic(f.attrs.keys())
    # m = json.loads(f.attrs['property_map'])
    # ic(m)
    test1 = f.create_group(exp.FLDR_NMS[0])
    # data_h = test1.create_dataset('hori', data=series)
    # ic(data_h.name)
    # ic(f[data_h.name][:10])
    for i in ['a', 'b', 'c']:
        g = f.create_group(i)
        g.create_dataset('i', data=np.arange(2))
        g.create_dataset('ii', data=np.arange(3))
    ic([nm for nm in f])
    ic(f['a'].keys())
    ic(f['a']['i'])
    arr = f['a']['i'][:]
    ic(arr)
