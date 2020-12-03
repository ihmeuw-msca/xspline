"""
Function Module
"""
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from xspline.fullfun import FullFunction
from xspline.interval import Interval
from xspline.funutils import check_fun_input, taylor_term


@dataclass
class ConstFunction(FullFunction):
    const: float = 1.0

    def __call__(self, data: Iterable, order: int = 0) -> np.ndarray:
        data, order = check_fun_input(data, order)
        if self.const == 0 or order > 0:
            val = np.zeros(data.shape[-1])
        else:
            val = self.const*taylor_term(data, -order)
        return val


@dataclass
class IndicatorFunction(FullFunction):

    def __post_init__(self):
        ldomain, rdomain = ~self.domain
        const_funs = [ConstFunction(const=1.0, domain=self.domain)]
        if ldomain is not None:
            const_funs.insert(0, ConstFunction(const=0.0, domain=ldomain))
        if rdomain is not None:
            const_funs.append(ConstFunction(const=0.0, domain=rdomain))
        self.fun = sum(const_funs)
