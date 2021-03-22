import json

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
    )
)

if __name__ == "__main__":
    fl_nm = 'config.json'
    with open(fl_nm, 'w') as f:
        json.dump(config, f)
