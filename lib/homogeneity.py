import numpy as np
import tqdm as tq
from hyppo.ksample import KSample

from .rank import sample as rnksample
from .rank import mthddecor as rnkdecorator
from .utils import FLOAT_TYPE

CALC_ACCURACY = 0.01


class HRTest(KSample):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def cdf(self, x, dim, *sizes, **kwargs):
        if "acc" in kwargs:
            acc = min(CALC_ACCURACY, kwargs["acc"])
        else:
            acc = CALC_ACCURACY
        iteration_num = int(2 * (1 / 4) / acc**2)
        
        proba = 0.
        for i in tq.tqdm(range(iteration_num)):
            rank_smpl = [rnksample(dim, size) for size in sizes]
            stat = self.statistic(*rank_smpl)
            if stat <= x:
                proba += 1.
        proba /= iteration_num
        return proba
    
    @rnkdecorator
    def statistic(self, *samples):
        return super().statistic(*samples)

    @rnkdecorator
    def test(self, *samples):
        return super().test(*samples)
