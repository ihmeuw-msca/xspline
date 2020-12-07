"""
Function Module
"""
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import lagrange

from xspline.fullfun import FullFunction
from xspline.funutils import check_fun_input, taylor_term, shift_poly
from xspline.interval import Interval


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
        self.fun = ConstFunction(const=1.0, domain=self.domain)
        if ldomain is not None:
            self.fun = ConstFunction(const=0.0, domain=ldomain) + self.fun
        if rdomain is not None:
            self.fun = self.fun + ConstFunction(const=0.0, domain=rdomain)


@dataclass
class PolyFunction(FullFunction):
    coefs: Iterable = (1.0,)

    def __post_init__(self):
        self.coefs = np.asarray(self.coefs)

    @property
    def degree(self) -> int:
        return len(self.coefs) - 1

    @classmethod
    def from_lagrange(cls, points: Iterable, weights: Iterable, **kwargs) -> "PolyFunction":
        kwargs.update({"coefs": lagrange(points, weights).coef[::-1]})
        return cls(**kwargs)

    @classmethod
    def from_taylor(cls, point: float, fun_ders: Iterable, **kwargs) -> "PolyFunction":
        coefs = [f/np.math.factorial(i) for i, f in enumerate(fun_ders)]
        kwargs.update({"coefs": shift_poly(coefs, -point)[0]})
        return cls(**kwargs)

    def __call__(self, data: Iterable, order: int = 0) -> np.ndarray:
        data, order = check_fun_input(data, order)
        if order == 0:
            val = np.polyval(self.coefs[::-1], data[-1])
        elif order > 0:
            val = np.polyval(np.polyder(self.coefs[::-1], order), data[-1])
        else:
            val = np.array(list(map(
                lambda coefs, d: np.polyval(np.polyint(coefs[::-1], -order), d),
                shift_poly(self.coefs, data[0]),
                data[1] - data[0]
            )))
        return val
