import numpy as np

import matplotlib.pyplot as plt

from icecream import ic

from vib_export import VibExportFEMTO


if __name__ == '__main__':
    exp = VibExportFEMTO()
    ic(exp.NUMS_MESR)
    idx_brg = 0
    vals = exp.get_single_measurement(0, idx_brg).to_numpy()
    ic(vals)
    vals_hori = vals[:, -2]
    vals_vert = vals[:, -1]
    l = vals_hori.shape[0]
    ic(vals_hori, vals_vert)

    plt.figure(figsize=(18, 6))
    plt.plot(np.arange(l), vals_hori, label='Horizontal acceleration')
    plt.plot(np.arange(l), vals_vert, label='Vertical acceleration')
    plt.legend(loc=0)
    plt.savefig('Explore single measurement.png', dpi=300)
    plt.show()

    ic(np.max(vals_hori) - np.min(vals_hori))
    ic(np.max(vals_vert) - np.min(vals_vert))

