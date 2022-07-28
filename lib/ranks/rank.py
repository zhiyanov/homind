import numpy as np
from scipy.spatial import distance_matrix as _dmt

from .assign import hungarian
from .assign import emd
from ..utils import FLOAT_TYPE

SHAPE_ERR = -1


def algosample(dim, size):
    pass

def randomsample(dim, size):
    radial = np.random.uniform(size=size).reshape((-1, 1))
    spherical = np.random.multivariate_normal(
            np.zeros(dim, dtype=FLOAT_TYPE),
            np.identity(dim, dtype=FLOAT_TYPE),
            size=size)
    distances = np.sqrt((spherical * spherical).sum(axis=1)).reshape((-1, 1))
    return spherical / distances * radial

def rankdata(data, ranks):
    if data.shape != ranks.shape:
        return SHAPE_ERR
    
    distance_matrix = _dmt(data, ranks)
    _, assignment = hungarian(distance_matrix)
    return ranks[assignment]

def decorator(is_method, is_algorithmic):
    def rank(*samples):
        samples = [np.array(smp, dtype=FLOAT_TYPE) for smp in samples] 
        dim = samples[0].shape[1]
        size = sum(smp.shape[0] for smp in samples)
        
        if is_algorithmic:
            ranks = algosample(dim, size)
        else:
            ranks = randomsample(dim, size)

        ssample = np.vstack(samples)
        ssample = rankdata(ssample, ranks)
        
        ranks = []
        start, end = 0, 0
        for smp in samples:
            end += smp.shape[0]
            ranks.append(ssample[start:end])
            start += smp.shape[0] 
        return ranks
        
    def dec(func):
        def function_wrapper(*samples, **kwargs):
            return func(*rank(*samples), **kwargs)
        
        def method_wrapper(self, *samples, **kwargs):
            return func(self, *rank(*samples), **kwargs)
        
        if is_method:
            return method_wrapper
        return function_wrapper

    return dec
