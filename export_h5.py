from vib_export import VibExp
from vib_extract import VibExtr

from icecream import ic

if __name__ == '__main__':
    exp = VibExp()
    extr = VibExtr()
    n = 0
    idx_brg = 0
    vals = exp.get_vib_values(n, idx_brg, acc='v')
    ic(vals)
    # rms_freq = extr.rms_freq(vals)
    # ic(rms_freq, rms_freq.shape)
    for nm, f in extr.D_PROP.items():
        ic(nm, f(vals))

