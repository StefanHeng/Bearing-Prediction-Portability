import json

onset_truth_hr = [  # Rough ground truth as measurement index, through manual inspection, by bearing test index
    7.2,
    2.29,
    2.36,
    2.09,
    1.36,
    3.98
]
onset_truth = [int(i * 6 * 60) for i in onset_truth_hr]  # Number of measurements per hour

config = dict(
    feat_disp_nms=dict(
        rms_time='RMS in time',
        range_time='Range in time',
        kurtosis='Kurtosis in time',
        skewness='Skewness in time',
        rms_freq='RMS in frequency',
        mean_freq='Mean in frequency',
        peak_freq='Frequency with max amplitude',
        rot_amp='Amplitude at rotating frequency'
    ),
    femto=dict(
        # Features fit for degradation detection
        degrading_indicators=['rms_time', 'range_time', 'kurtosis', 'skewness'],
        onset_truth=onset_truth
    )
)

if __name__ == "__main__":
    fl_nm = 'config.json'
    with open(fl_nm, 'w') as f:
        json.dump(config, f)
