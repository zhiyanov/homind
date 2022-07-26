import numpy as np

from lib.homogeneity import HRDist
from lib.utils import FLOAT_TYPE


def homogeneity_test_H1():
    first_size, second_size = 100, 100
    first_sample = np.random.multivariate_normal(
            np.zeros(2),
            np.array([[2, 1], [1, 2]]),
            size=first_size)
    second_sample = np.random.multivariate_normal(
            np.zeros(2),
            np.array([[1, 0], [0, 1]]),
            size=second_size)

    test = HRDist(indep_test="Dcorr")
    return test.statistic(first_sample, second_sample)

homogeneity_test_H1()
