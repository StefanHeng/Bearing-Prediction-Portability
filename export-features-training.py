"""
Actually export the FEMTO dataset into HDF5 format
"""

from icecream import ic

from vib_export import VibExportFEMTO

if __name__ == '__main__':
    exp = VibExportFEMTO()
    exp.export()
