import numpy as np
from scipy.spatial import distance_matrix as _dmt

from .assign import hungarian
from .assing import emd
from .utils import FLOAT_TYPE

SHAPE_ERR = -1


def decorator(func):
    def wrapper(*samples): 
        samples = [np.array(sample, dtype=FLOAT_TYPE) for sample in samples)]  
        dim = samples[0].shape[1]
        size = sum(sample.shape[0] for sample in samples)
        
        ranks = spherical_sample(dim, size)
        ssample = np.vstack(samples)
        ssample = rank_data(ssample, ranks)
        
        ranked_samples = []
        start, end = 0, 0
        for sample in samples:
            end += sample.shape[0]
            ranked_samples.append(ssample[start:end])
            start += sample.shape[0]

        return func(ranked_samples)
    
    return wrapper
    

def rankdata(data, rank):
    if data.shape != rank.shape:
        return SHAPE_ERR
    
    distance_matrix = _dmt(data, rank)
    _, assignment = hungarian(distance_matrix)
    return rank[assignment]

def sample(dim, size):
    radial = np.random.uniform(size=size)
    spherical = np.random.normal(
            np.zeros(dim, dtype=FLOAT_TYPE),
            np.identity(dim, dtype=FLOAT_TYPE),
            dtype=FLOAT_TYPE) 
    return spherical * radial
