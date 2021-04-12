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

from icecream import ic
from copy import deepcopy

FEAT_DISP_NMS = dict(
    rms_time='RMS in time',
    range_time='Range in time',
    kurtosis='Kurtosis in time',
    skewness='Skewness in time',
    rms_freq='RMS in frequency',
    mean_freq='Mean in frequency',
    peak_freq='Frequency with max amplitude',
    rot_amp='Amplitude at rotating frequency'
)

# Respectively for 1-indexed [Test1 Bearing 3, Test1 Bearing 4, Test2 Bearing 1, Test3 Bearing 3]
onsets_hr_ims_prev = [330, 330, 130, 1023]  # Output of previous method
onsets_ims_prev = [i * 6 for i in onsets_hr_ims_prev]  # 10 minutes interval per measurement
features_ims = list(FEAT_DISP_NMS.keys())

bearings_failed_ims = {  # Bearing indices with observed failure for each test, 0-indexed
    0: [2, 3],  # Per README, 0-indexed bearings
    1: [0],
    2: [2]
}


def get_degrading_onsets_prev_ims():
    degrading_onsets_prev_ims = dict()
    i = 0
    for idx_tst in bearings_failed_ims:
        d = degrading_onsets_prev_ims[idx_tst] = dict()
        for idx_brg in bearings_failed_ims[idx_tst]:
            d[idx_brg] = onsets_ims_prev[i]
            i += 1
    return degrading_onsets_prev_ims


degrading_onsets_by_feature_detected_ims = {  # By my implementation
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
}


def get_degrading_onsets_detected_ims():
    degrading_onsets_detected_ims = dict()
    idxs_tst = [0, 2, 3]  # My test 1 ignored as it's ignored in previous project
    for idx_tst in idxs_tst:
        d = degrading_onsets_detected_ims[idx_tst] = dict()
        do = degrading_onsets_by_feature_detected_ims[idx_tst]
        for idx_brg in bearings_failed_ims[max(0, idx_tst - 1)]:
            d[idx_brg] = max(do[idx_brg].values())
    return degrading_onsets_detected_ims


num_tests_femto = 6
onset_truth_hr_femto = [  # Rough ground truth as measurement index, through manual inspection, by bearing test index
    4,
    1,
    0.75,
    0.75,
    1.3,
    4
]
onset_truth_femto = [int(i * 6 * 60) for i in onset_truth_hr_femto]  # Number of measurements per hour
features_femto = deepcopy(features_ims)
features_femto.pop()

degrading_onsets_by_feature_prev_femto = {
    0: {'kurtosis': 407, 'peak_freq': 354, 'range_time': 1404, 'skewness': 778},
    1: {'kurtosis': -1, 'peak_freq': 747, 'range_time': 824, 'skewness': 113},
    2: {'kurtosis': 833, 'peak_freq': 118, 'range_time': 825, 'skewness': 901},
    3: {'kurtosis': 741, 'peak_freq': 196, 'range_time': 222, 'skewness': 743},
    4: {'kurtosis': 476, 'peak_freq': 501, 'range_time': 469, 'skewness': 476},
    5: {'kurtosis': 1430, 'peak_freq': 74, 'range_time': 1425, 'skewness': 115}
}
degrading_onsets_by_feature_new_femto = {
    0: {'mean_freq': 440, 'peak_freq': 354, 'range_time': 1404},
    1: {'mean_freq': 115, 'peak_freq': 747, 'range_time': 824},
    2: {'mean_freq': 112, 'peak_freq': 118, 'range_time': 825},
    3: {'mean_freq': 160, 'peak_freq': 196, 'range_time': 222},
    4: {'mean_freq': 473, 'peak_freq': 501, 'range_time': 469},
    5: {'mean_freq': 175, 'peak_freq': 74, 'range_time': 1425}
}


def get_degrading_onsets_femto(onsets):
    return {k: max(v.values()) for k, v in onsets.items()}
    # d = dict()
    # for idx_tst in range(num_tests_femto):
    #     d[idx_tst] = max(onsets[idx_tst].values())
    # return d


config = dict(
    feature_display_names=FEAT_DISP_NMS,
    ims=dict(
        sample_rate=20_000,
        num_bearings=4,
        # Number of measurements in tests 1st, 2nd, 4th
        # The first row for features in the time domain was removed in previous project
        num_measurements=[2_156, 984, 6_324],
        num_samples=20_480,  # Number of data points in each measurement file

        # Learned characteristics on the dataset
        # Features fit for detecting degradation onset, selected by previous project
        features=features_ims,
        num_features=len(features_ims),
        degrading_indicators=['kurtosis', 'skewness', 'range_time', 'peak_freq'],
        degrading_hyperparameters=dict(  # Selected for final output in previous project
            kurtosis=dict(sz_base=100, sz_window=30, z=3),
            skewness=dict(sz_base=100, sz_window=15, z=3),
            range_time=dict(sz_base=100, sz_window=30, z=5),
            peak_freq=dict(sz_base=100, sz_window=25, z=5)  # Only the upperbound is checked for detection
        ),
        bearings_failed=bearings_failed_ims,
        degrading_onsets_prev=get_degrading_onsets_prev_ims(),
        # degrading_onsets_prev={   # Output quoted from previous project
        #     0: {
        #         2: onsets_ims_prev[0],
        #         3: onsets_ims_prev[1]
        #     },
        #     1: {0: onsets_ims_prev[2]},
        #     2: {2: onsets_ims_prev[3]},
        # },
        # My output with previously tuned hyperparameters
        degrading_onsets_by_feature_detected=degrading_onsets_by_feature_detected_ims,
        degrading_onsets_detected=get_degrading_onsets_detected_ims()
        # degrading_onsets_detected={
        #     0: {2: 1816, 3: 1457},
        #     2: {0: 710},
        #     3: {2: 6235}
        # }
    ),
    femto=dict(
        sample_rate=25_600,
        num_tests=num_tests_femto,
        # Number of roll bearings in training data, each corresponding to a run-to-failure trial
        num_samples=25_600 // 10,  # 0.1 second of data
        num_measurements=[2803, 871, 911, 797, 515, 1637],

        # Learned characteristics on the dataset
        # Features fit for degradation detection
        features=features_femto,
        num_features=len(features_femto),
        onset_truth=onset_truth_femto,
        degrading_hyperparameters_prev=dict(  # Degrading indicators selected in previous project tuned for FEMTO
            kurtosis=dict(sz_base=60, sz_window=15, z=3),
            skewness=dict(sz_base=60, sz_window=15, z=2),
            range_time=dict(sz_base=60, sz_window=25, z=2),
            peak_freq=dict(sz_base=60, sz_window=45, z=1.5)
        ),
        degrading_onsets_by_feature_prev=degrading_onsets_by_feature_prev_femto,
        degrading_onsets_prev=get_degrading_onsets_femto(degrading_onsets_by_feature_prev_femto),
        # New degrading indicators selected for FEMTO based on the same criteria as previous project
        degrading_indicators_new=['range_time', 'mean_freq', 'peak_freq'],
        degrading_hyperparameters_new=dict(
            range_time=dict(sz_base=60, sz_window=25, z=2),
            mean_freq=dict(sz_base=60, sz_window=30, z=2),
            peak_freq=dict(sz_base=60, sz_window=45, z=1.5)
        ),
        degrading_onsets_by_feature_new=degrading_onsets_by_feature_new_femto,
        degrading_onsets_new=get_degrading_onsets_femto(degrading_onsets_by_feature_new_femto),
        degrading_onsets_new_subset={k: sorted(v.values())[-2]  # Degradation detected in n-1 of n indicators
                                     for k, v, in degrading_onsets_by_feature_new_femto.items()}
    )
)

if __name__ == "__main__":
    import json

    fl_nm = 'config.json'
    ic(config)
    with open(fl_nm, 'w') as f:
        json.dump(config, f, indent=4)
