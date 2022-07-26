import numpy as np
import matplotlib.pyplot as plt

from lib.homogeneity import HRTest
from lib.utils import FLOAT_TYPE
import lib.rank as rank


def homogeneity_test_normal_H1():
    first_size, second_size = 100, 100
    first_sample = np.random.multivariate_normal(
            np.zeros(2),
            np.array([[1, 0.5], [0.5, 1]]),
            size=first_size)
    second_sample = np.random.multivariate_normal(
            np.zeros(2),
            np.array([[1, 0], [0, 1]]),
            size=second_size)
    
    ranks = rank.sample(2, first_size + second_size)
    ranked = rank.rankdata(
            np.vstack([first_sample, second_sample]),
            ranks)
    ranked = [ranked[0:first_size], ranked[first_size:]]
    
    # plt.scatter(first_sample[:, 0], first_sample[:, 1], color="red", marker="o")
    plt.scatter(ranked[0][:, 0], ranked[0][:, 1], color="red", marker="v")
    # plt.scatter(second_sample[:, 0], second_sample[:, 1], color="blue", marker="o")
    plt.scatter(ranked[1][:, 0], ranked[1][:, 1], color="blue", marker="v")
    plt.axes().set_aspect("equal", adjustable='box')
    plt.savefig("./picture.png")


    test = HRTest(indep_test="Dcorr")

    stat = test.statistic(first_sample, second_sample)
    pvalue = test.cdf(stat, 2, first_size, second_size)
    return stat, pvalue

def homogeneity_test_circle_H1():
    first_size, second_size = 100, 100
    first_sample = np.random.multivariate_normal(
            np.zeros(2),
            np.array([[0.25, 0], [0, 0.25]]),
            size=first_size)

    second_sample = np.random.multivariate_normal(
            np.zeros(2),
            np.array([[1, 0], [0, 1]]),
            size=second_size)
    
    ranks = rank.sample(2, first_size + second_size)
    ranked = rank.rankdata(
            np.vstack([first_sample, second_sample]),
            ranks)
    ranked = [ranked[0:first_size], ranked[first_size:]]
    
    # plt.scatter(first_sample[:, 0], first_sample[:, 1], color="red", marker="o")
    plt.scatter(ranked[0][:, 0], ranked[0][:, 1], color="red", marker="v")
    # plt.scatter(second_sample[:, 0], second_sample[:, 1], color="blue", marker="o")
    plt.scatter(ranked[1][:, 0], ranked[1][:, 1], color="blue", marker="v")
    plt.axes().set_aspect("equal", adjustable='box')
    plt.savefig("./picture.png")

    test = HRTest(indep_test="Dcorr")

    stat = test.statistic(first_sample, second_sample)
    pvalue = test.cdf(stat, 2, first_size, second_size)
    return stat, pvalue
