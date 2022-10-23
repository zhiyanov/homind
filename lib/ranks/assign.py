import numpy as np

from ot.lp import emd as _emd
from scipy.optimize import linear_sum_assignment as _lsm

from ..utils import format_float


def emd(distance_matrix):
    distributions = [np.ones(distance_matrix.shape[i], dtype=DATA_TYPE) \
        / distance_matrix.shape[i] for i in range(2)
    ]
    distance_matrix = format_float(distance_matrix)
    
    plan =  _emd(*distributions, distance_matrix)
    start = np.arrange(distance_matrix.shape[0])
    end = np.argmax(plan, axis=1)
    
    return start, end

def hungarian(distance_matrix):
    distance_matrix = format_float(distance_matrix)
    start, end = _lsm(distance_matrix)

    return start, end 
