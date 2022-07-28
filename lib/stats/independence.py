import numpy as np
import tqdm as tq

from hyppo.independence import MaxMargin
from hyppo.independence import KMERF
from hyppo.independence import MGC

from ..ranks.rank import randomsample as rank_sample
from ..ranks.rank import decorator as rank_decorator
from ..utils import FLOAT_TYPE

CALC_ACCURACY = 0.01


class RankedMaxMargin(MaxMargin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def cdf(self, x, dim, *sizes, **kwargs):
        if "acc" in kwargs:
            acc = min(CALC_ACCURACY, kwargs["acc"])
        else:
            acc = CALC_ACCURACY
        iteration_num = int(4 * (1 / 4) / acc**2)
        
        proba = 0.
        for i in tq.tqdm(range(iteration_num)):
            rank_smpl = [rank_sample(dim, size) for size in sizes]
            stat = self.statistic(*rank_smpl)
            if stat <= x:
                proba += 1.
        proba /= iteration_num
        return proba
    
    @rank_decorator(True, False)
    def statistic(self, x, y):
        return super().statistic(x, y)

    @rank_decorator(True, False)
    def test(self, x, y, **kwargs):
        return super().test(x, y, **kwargs)

class RankedKMERF(KMERF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def cdf(self, x, dim, *sizes, **kwargs):
        if "acc" in kwargs:
            acc = min(CALC_ACCURACY, kwargs["acc"])
        else:
            acc = CALC_ACCURACY
        iteration_num = int(4 * (1 / 4) / acc**2)
        
        proba = 0.
        for i in tq.tqdm(range(iteration_num)):
            rank_smpl = [rank_sample(dim, size) for size in sizes]
            stat = self.statistic(*rank_smpl)
            if stat <= x:
                proba += 1.
        proba /= iteration_num
        return proba
    
    @rank_decorator(True, False)
    def statistic(self, x, y):
        return super().statistic(x, y)

    @rank_decorator(True, False)
    def test(self, x, y, **kwargs):
        return super().test(x, y, **kwargs)

class RankedMGC(MGC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def cdf(self, x, dim, *sizes, **kwargs):
        if "acc" in kwargs:
            acc = min(CALC_ACCURACY, kwargs["acc"])
        else:
            acc = CALC_ACCURACY
        iteration_num = int(4 * (1 / 4) / acc**2)
        
        proba = 0.
        for i in tq.tqdm(range(iteration_num)):
            rank_smpl = [rank_sample(dim, size) for size in sizes]
            stat = self.statistic(*rank_smpl)
            if stat <= x:
                proba += 1.
        proba /= iteration_num
        return proba
    
    @rank_decorator(True, False)
    def statistic(self, x, y):
        return super().statistic(x, y)

    @rank_decorator(True, False)
    def test(self, x, y, **kwargs):
        return super().test(x, y, **kwargs)
