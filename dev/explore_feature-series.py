import numpy as np
import matplotlib.pyplot as plt

from icecream import ic

from vib_export import VibExport
from vib_extract import VibExtract

if __name__ == '__main__':
    exp = VibExport()
    extr = VibExtract()
    # ic(exp.NUMS_FL)
    n = 0
    idx_brg = 0
    series = exp.get_feature_series(idx_brg, extr.rms_time)
    l = exp.NUMS_FL[idx_brg]
    ic(series)

    plt.figure(figsize=(18, 6))
    plt.plot(np.arange(l), series)
    plt.show()
