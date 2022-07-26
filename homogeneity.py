import numpy as np
from hyppo.ksample import KSample

from .rank import sample as rnksample
from .rank import decorator as rnkdecorator
from .utils import FLOAT_TYPE

CALC_ACCURACY = 0.01


class HRDist(KSample):
    def cdf(self, x, dim, *sizes, **kwargs):
        if "acc" in kwargs:
            acc = min(CALC_ACCURACY, kwargs["acc"])
        else:
            acc = CALC_ACCURACY
        iteration_num = 3 * int(1 / acc**2)
        
        proba = 0.
        for i in range(iteration_num):
            rank_smpl = [rnksample(dim, size) for size in sizes]
            stat = self.statistic(*rank_smpl)
            if stat <= x:
                proba += 1.
        proba /= iteration_num
        return proba
    
    @rnkdecorator
    def statistic(self, *samples):
        return super().statistic(samples)

    @rnkdecorator
    def test(self, *samples):
        return super().test(samples)
