from random import randint

DATA_PATH = '/Users/stefanh/Documents/UMich/Career/Predictive Maintenance/Dataset/FEMTOBearingDataSet/Learning_set/'
DATA_PATH_EG = f'{DATA_PATH}Bearing1_1/acc_{1:05}.csv'
data = '/Users/stefanh/Documents/UMich/Career/Predictive Maintenance/Prediction Model Portability/data/'
H5FEAT_PATH = f'{data}femto_features_train.hdf5'  # For features already extracted

# The features extracted, not the IMS raw dataset
FEAT_PATH_IMS = '/Users/stefanh/Documents/UMich/Career/Predictive Maintenance/Previous Work/' \
                'Degradation Onset Detection/MATLAB/features extracted, all rows/'
H5FEAT_PATH_IMS = f'{data}ims_features.hdf5'


def DEBUG_get_rand_range(n=10000, end=-1):
    """ Returns random range of length `N`, for debugging purposes only """
    strt = randint(0, end - n)
    return strt, strt + n
