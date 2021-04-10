"""
Actually export the FEMTO dataset into HDF5 format
"""

from icecream import ic

from vib_export_femto import VibExportFemto

if __name__ == '__main__':
    exp = VibExportFemto()
    exp.export()
