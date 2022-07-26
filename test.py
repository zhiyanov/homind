import numpy as np

from lib.homogeneity import HRDist
from lib.utils import FLOAT_TYPE


def homogeneity_test_H1():
    first_size, second_size = 100
    first_samle = np.random.multivariate_normal(
            np.zeros(2),
            np.array([[2, 1], [1, 2]]),
            size=first_size,
            dtype=FLOAT_TYPE)
    second_samle = np.random.multivariate_normal(
            np.zeros(2),
            np.array([[1, 0], [0, 1]]),
            size=second_size,
            dtype=FLOAT_TYPE)

    return HRDist(indep_test="Dcorr").test(first_samle, second_samle)

homogeneity_test_H1()
