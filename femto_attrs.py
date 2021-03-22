"""
Each acceleration file has 0.1s of data every 10 second.

References
----------
.. [1] A. Saxena and K. Goebel (2008). "PHM08 Challenge Data Set", NASA Ames Prognostics Data Repository
        (http://ti.arc.nasa.gov/project/prognostic-data-repository), NASA Ames Research Center, Moffett Field, CA
"""
SPL_RT = 25_600  # Sample rate
NUM_BRG = 6  # Number of roll bearings in training data
N_SPL = SPL_RT // 10  # 0.1 second of data
