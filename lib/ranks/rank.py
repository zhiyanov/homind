import numpy as np
from scipy.spatial import distance_matrix as _dmt

from .assign import hungarian
from .assign import emd
from ..utils import FLOAT_TYPE

SHAPE_ERR = -1


def algosample(dim, size):
    pass

def randomgenerator(dim, size, seed=None):
    rng = np.random.default_rng(seed)
    while True:
        radial = rng.uniform(size=size).reshape((-1, 1))
        spherical = rng.multivariate_normal(
                np.zeros(dim, dtype=FLOAT_TYPE),
                np.identity(dim, dtype=FLOAT_TYPE),
                size=size)
        distances = np.sqrt((spherical * spherical).sum(axis=1)).reshape((-1, 1))
        yield spherical / distances * radial

def randomsample(dim, size, seed=None):
    rng = np.random.default_rng(seed)
    radial = rng.uniform(size=size).reshape((-1, 1))
    spherical = rng.multivariate_normal(
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

def rank(samples, ranks):
    lengths = [len(smp) for smp in samples]

    samples = [np.array(smp, dtype=FLOAT_TYPE) for smp in samples]
    samples = np.vstack(samples)
    
    ranks = [np.array(rnk, dtype=FLOAT_TYPE) for rnk in ranks]
    ranks = np.vstack(ranks)

    rsample = rankdata(samples, ranks)
    
    ranks = []
    start, end = 0, 0
    for length in lengths:
        end += length
        ranks.append(rsample[start:end])
        start += length
    return ranks

def decorator(seed=None, method=False):
    def rank(*samples, seed=None):
        samples = [np.array(smp, dtype=FLOAT_TYPE) for smp in samples] 
        dim = samples[0].shape[1]
        size = sum(smp.shape[0] for smp in samples)
        
        if seed is None:
            ranks = algosample(dim, size)
        else:
            ranks = randomsample(dim, size, seed=seed)

        rsample = np.vstack(samples)
        rsample = rankdata(rsample, ranks)
        
        ranks = []
        start, end = 0, 0
        for smp in samples:
            end += smp.shape[0]
            ranks.append(rsample[start:end])
            start += smp.shape[0] 
        return ranks

    def dec(func):
        def function_wrapper(*samples, **kwargs):
            return func(*rank(*samples, seed=seed), **kwargs)
        
        def method_wrapper(self, *samples, **kwargs):
            return func(self, *rank(*samples, seed=seed), **kwargs)
        
        if method:
            return method_wrapper
        return function_wrapper

    return dec
