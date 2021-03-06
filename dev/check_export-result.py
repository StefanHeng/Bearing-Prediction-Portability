import numpy as np

from icecream import ic

from vib_export import VibExport
from vib_extract import VibExtract
from vib_record import VibRecord

if __name__ == '__main__':
    rec = VibRecord()
    series = rec.get_feature_series(0)
    ic(series, series.shape)

    exp = VibExport()
    extr = VibExtract
    assert np.array_equal(series, exp.get_feature_series(0, extr.rms_time))
