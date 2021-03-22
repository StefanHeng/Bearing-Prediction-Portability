"""
Each measurement has 0.1s of data every 10 minute.

References
----------
.. [1] J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services (2007).
        IMS, University of Cincinnati. "Bearing Data Set", NASA Ames Prognostics Data Repository
        (http://ti.arc.nasa.gov/project/prognostic-data-repository), NASA Ames Research Center, Moffett Field, CA
"""

SPL_RT = 20_000  # Sample rate
# Number of measurements in tests 1st, 2nd, 4th
NUMS_MESR = [2_156, 984, 6_324]  # The first row for features in the time domain was removed in previous project
NUM_BRG = 4
N_SPL = 20_480  # Number of data points in each measurement file
