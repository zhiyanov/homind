import numpy as np

from lib.homogeneity import HRTest
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

    test = HRTest(indep_test="Dcorr")

    stat = test.statistic(first_sample, second_sample)
    pvalue = test.cdf(stat, 2, first_size, second_size)
    return stat, pvalue

print(homogeneity_test_H1())
