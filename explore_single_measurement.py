import numpy as np

import matplotlib.pyplot as plt

from icecream import ic

from vib_export import VibExp


if __name__ == '__main__':
    exp = VibExp()
    ic(exp.NUMS_FL)
    idx_brg = 0
    vals = exp.get_vib_values(0, idx_brg)
    ic(vals)
    vals_hori = vals[:, -2]
    vals_vert = vals[:, -1]
    l = vals_hori.shape[0]
    ic(vals_hori, vals_vert)

    plt.figure(figsize=(18, 6))
    plt.plot(np.arange(l), vals_hori)
    plt.plot(np.arange(l), vals_vert)
    plt.show()

    ic(np.max(vals_hori) - np.min(vals_hori))
    ic(np.max(vals_vert) - np.min(vals_vert))

