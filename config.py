"""
The IMS dataset
Measurements taken every 10 minute.

References
----------
.. [1] J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services (2007).
        IMS, University of Cincinnati. "Bearing Data Set", NASA Ames Prognostics Data Repository
        (http://ti.arc.nasa.gov/project/prognostic-data-repository), NASA Ames Research Center, Moffett Field, CA


The FEMTO dataset
Each acceleration file has 0.1s of data every 10 second.

References
----------
.. [1] A. Saxena and K. Goebel (2008). "PHM08 Challenge Data Set", NASA Ames Prognostics Data Repository
        (http://ti.arc.nasa.gov/project/prognostic-data-repository), NASA Ames Research Center, Moffett Field, CA
"""

# Respectively for 1-indexed [Test1 Bearing 3, Test1 Bearing 4, Test2 Bearing 1, Test3 Bearing 3]
onsets_hr_ims_prev = [330, 330, 130, 1023]  # Output of previous method
onsets_ims_prev = [i * 6 for i in onsets_hr_ims_prev]  # 10 minutes interval per measurement

onset_truth_hr_femto = [  # Rough ground truth as measurement index, through manual inspection, by bearing test index
    7.2,
    2.29,
    2.36,
    2.09,
    1.36,
    3.98
]
onset_truth_femto = [int(i * 6 * 60) for i in onset_truth_hr_femto]  # Number of measurements per hour

config = dict(
    feature_display_names=dict(
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
        sample_rate=25_600,
        num_bearings=6,  # Number of roll bearings in training data, each corresponding to a run-to-failure trial
        num_samples=25_600 // 10,  # 0.1 second of data

        # Learned characteristics on the dataset
        # Features fit for degradation detection
        degrading_indicators=['rms_time', 'range_time', 'kurtosis', 'skewness'],
        onset_truth=onset_truth_femto
    ),
    ims=dict(
        sample_rate=20_000,
        num_bearings=4,
        # Number of measurements in tests 1st, 2nd, 4th
        # The first row for features in the time domain was removed in previous project
        num_measurements=[2_156, 984, 6_324],
        num_samples=20_480,  # Number of data points in each measurement file
        bearings_failed={  # Bearing indices with observed failure for each test, 0-indexed
            0: [2, 3],  # Per README, 0-indexed bearings
            1: [0],
            2: [2]
        },

        # Learned characteristics on the dataset
        # Features fit for detecting degradation onset, selected by previous project
        degrading_indicators=['kurtosis', 'skewness', 'range_time', 'peak_freq'],
        prev_hyperparameters=dict(  # Selected for final output in previous project
            kurtosis=dict(sz_base=100, sz_window=30, z=3),
            skewness=dict(sz_base=100, sz_window=15, z=3),
            range_time=dict(sz_base=100, sz_window=30, z=5),
            peak_freq=dict(sz_base=100, sz_window=25, z=5)  # Only the upperbound is checked for detection
        ),
        prev_onsets={
            0: {
                2: onsets_ims_prev[0],
                3: onsets_ims_prev[1]
            },
            1: {0: onsets_ims_prev[2]},
            2: {2: onsets_ims_prev[3]},
        },
        # My output with previously tuned hyperparameters
        prev_onsets_detected_by_feature={
            0: {
                2: {'kurtosis': 1797, 'peak_freq': 170, 'range_time': 1814, 'skewness': 1816},
                3: {'kurtosis': 233, 'peak_freq': 1348, 'range_time': 1425, 'skewness': 1457}
            },
            1: {  # 2nd dimension of test 1
                2: {'kurtosis': 1805, 'peak_freq': -1, 'range_time': 1818, 'skewness': -1},
                3: {'kurtosis': 1599, 'peak_freq': 100, 'range_time': 1600, 'skewness': -1}
            },
            2: {
                0: {'kurtosis': 630, 'peak_freq': 647, 'range_time': 682, 'skewness': 710}
            },
            3: {
                2: {'kurtosis': 6140, 'peak_freq': 6235, 'range_time': 6143, 'skewness': 6076}}
        },
        prev_onsets_detected={
            0: {2: 1816, 3: 1457},
            2: {0: 710},
            3: {2: 6235}
        }
    )
)

if __name__ == "__main__":
    import json
    fl_nm = 'config.json'
    with open(fl_nm, 'w') as f:
        json.dump(config, f, indent=4)
