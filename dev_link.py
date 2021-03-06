from random import randint

DATA_PATH = '/Users/stefanh/Documents/UMich/Career/Predictive Maintenance/Dataset/FEMTOBearingDataSet/Learning_set/'
DATA_PATH_EG = f'{DATA_PATH}Bearing1_1/acc_{1:05}.csv'
FEAT_PATH = '/Users/stefanh/Documents/UMich/Career/Predictive Maintenance/' \
            'Prediction Model Portability/data/femto_features_train.hdf5'  # For features already extracted


def DEBUG_get_rand_range(n=10000, end=-1):
    """ Returns random range of length `N`, for debugging purposes only """
    strt = randint(0, end - n)
    return strt, strt + n
