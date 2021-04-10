import numpy as np

from icecream import ic

from vib_export_femto import VibExportFemto
from vib_extract import VibExtract
from vib_record_femto import VibRecordFemto

if __name__ == '__main__':
    rec = VibRecordFemto()
    series = rec.get_feature_series(0)
    ic(series, series.shape)

    exp = VibExportFemto()
    extr = VibExtract
    assert np.array_equal(series, exp.get_feature_series(0, extr.rms_time))
