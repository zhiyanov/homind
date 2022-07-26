import numpy as np
from scipy.spatial import distance_matrix as _dmt

from .assign import hungarian
from .assign import emd
from .utils import FLOAT_TYPE

SHAPE_ERR = -1


def sample(dim, size):
    radial = np.random.uniform(size=size).reshape((-1, 1))
    spherical = np.random.multivariate_normal(
            np.zeros(dim, dtype=FLOAT_TYPE),
            np.identity(dim, dtype=FLOAT_TYPE),
            size=size)
    return spherical * radial

def rankdata(data, rank):
    if data.shape != rank.shape:
        return SHAPE_ERR
    
    distance_matrix = _dmt(data, rank)
    _, assignment = hungarian(distance_matrix)
    return rank[assignment]

def mthddecor(func):
    def wrapper(self, *samples): 
        samples = [np.array(smp, dtype=FLOAT_TYPE) for smp in samples]  
        dim = samples[0].shape[1]
        size = sum(smp.shape[0] for smp in samples)
        
        ranks = sample(dim, size)
        ssample = np.vstack(samples)
        ssample = rankdata(ssample, ranks)
        
        ranked = []
        start, end = 0, 0
        for smp in samples:
            end += smp.shape[0]
            ranked.append(ssample[start:end])
            start += smp.shape[0]

        return func(self, *ranked)
    
    return wrapper    
