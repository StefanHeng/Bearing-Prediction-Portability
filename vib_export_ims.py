class VibExport:
    """ Handles reading IMS dataset feature extraction output in CSV and exporting to h5py data record,
    with the following properties: RMS_time, Range_time, Kurtosis, Skewness, Range_freq, Peak_freq, Mean_freq,

    Each roll bearing's measurement recorded at a time will be 1-dimensional.
    """

