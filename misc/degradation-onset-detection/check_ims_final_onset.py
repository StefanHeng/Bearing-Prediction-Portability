""" Re-run the detection algorithm with tuned hyperparameters, check the final onset output
"""

import os

from util import *

os.chdir('../..')


if __name__ == '__main__':
    onsets = config('ims.prev_onsets')
    onsets_d = config('ims.prev_onsets_detected_by_feature')
    # My idx_tst with 1 makes up the other dimension, which seems wasn't considered in previous project output
    idxs_tst = [0, 2, 3]
    ic(onsets, onsets_d)

    for idx_tst in idxs_tst:
        d = onsets_d[str(idx_tst)]
        for idx_brg in d:
            idx_tst = idx_tst+1
            onset = max(d[idx_brg].values())
            idx_brg = int(idx_brg)+1
            ic(idx_tst, idx_brg, onset)

