from abc import ABC, abstractmethod
from multiprocessing import Process, Queue

import numpy as np
import tqdm

from .rank import randomsample as _randomsample
from .rank import randomgenerator as _randomgenerator
from .rank import rank as _rank

from ..utils import INT_TYPE, FLOAT_TYPE

DIMENSION_ERR = -1
DISTRIBUTION_ERR = -2


class Generator(ABC):
    def __init__(self, statistic, dimension, *sizes, seed=None):
        self.st = statistic
        self.dimension = dimension
        self.sizes = np.array(sizes, dtype=INT_TYPE)
        
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.ranks = _randomsample(
                self.dimension,
                self.sizes.sum(),
                seed=self.seed)

        self.dst = None

    def statistic(self, *samples, **kwargs):
        ranks = _rank(samples, self.ranks)
        return self.st(*ranks, **kwargs)

    def test(self, *samples, **kwargs):
        if self.dst is None:
            return DISTRIBUTION_ERR

        stat = self.statistic(*samples, **kwargs)
        pvalue = 1 - self.cdf(stat)
        return stat, pvalue
    
    @abstractmethod
    def sample(self, seed=None, **kwargs):
        pass
     
    def generate(self, size, seed=None, queue=None, **kwargs):
        sampler = self.sample(seed=seed, **kwargs)
        result = np.zeros(size, dtype=FLOAT_TYPE)
        for i in tqdm.tqdm(range(size)):
            result[i] = next(sampler)

        if not (queue is None):
            queue.put(result)

        return result

    def distribution(self, size, seeds=None, **kwargs):
        if not seeds:
            seeds = [self.seed]
        
        procs = []
        queue = Queue()
        psize = size // len(seeds)
        for i, seed in enumerate(seeds):
            if i == len(seeds) - 1:
                psize += size % len(seeds)
            
            proc = Process(
                target=self.generate,
                args=(psize, seed, queue),
                kwargs=kwargs
            )
            proc.start()
            procs.append(proc)

        for i in range(len(procs)):
            procs[i].join()
        
        result = []
        while not queue.empty():
            result.append(queue.get())
        
        result = np.hstack(result)
        result.sort()
        self.dst = result
        return self.dst

    def cdf(self, quantile):
        if self.dst is None:
            return DISTRIBUTION_ERR

        index = np.searchsorted(
                self.dst,
                quantile,
                side="right")
        return index / len(self.dst)

class Permuter(Generator):
    def sample(self, seed=None, **kwargs):
        if seed is None:
            rng = self.rng
        else:
            rng = np.random.default_rng(seed)

        while True:
            permutation = rng.permutation(len(self.ranks))
            ranks = self.ranks[permutation]
            ranks = np.split(ranks, np.cumsum(self.sizes)[:-1])
            yield self.st(*ranks, **kwargs)
            
class Limiter(Generator):
    def sample(self, seed=None, **kwargs):
        generator = _randomgenerator(
            self.dimension, self.sizes.sum(),
            seed=seed)

        while True:
            ranks = next(generator)
            ranks = np.split(ranks, np.cumsum(self.sizes)[:-1])
            yield self.st(*ranks, **kwargs)
