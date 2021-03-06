"""
Actually export the FEMTO dataset into h5py
"""

from icecream import ic

from vib_export import VibExport

if __name__ == '__main__':
    exp = VibExport()
    exp.export()
