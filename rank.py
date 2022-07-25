import numpy as np
from scipy.spatial import distance_matrix as _dmt

from .assign import hungarian
from .assing import emd
from .utils import FLOAT_TYPE

SHAPE_ERR = -1


def rank_data(data, rank):
    if data.shape != rank.shape:
        return SHAPE_ERR
    
    distance_matrix = _dmt(data, rank)
    _, assignment = hungarian(distance_matrix)
    return rank[assignment]

def spehrical_sample(size, dim):
    radial = np.random.uniform(size=size)
    spherical = np.random.normal(
            np.zeros(dim, dtype=FLOAT_TYPE),
            np.identity(dim, dtype=FLOAT_TYPE),
            dtype=FLOAT_TYPE) 
    return spherical * radial
