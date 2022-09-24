from math import factorial

import numpy as np

from xspline.typing import IndiParams, NDArray
from xspline.xfunction import BundleXFunction


def indi_val(params: IndiParams, x: NDArray) -> NDArray:
    # lower and upper bounds
    lb, ub = params

    val = np.zeros(x.size, dtype=x.dtype)
    ind = (x > lb[0]) & (x < ub[0])
    if lb[1]:
        ind = ind | (x == lb[0])
    if ub[1]:
        ind = ind | (x == ub[0])
    val[ind] = 1.0
    return val


def indi_der(params: IndiParams, x: NDArray, order: int) -> NDArray:
    return np.zeros(x.size, dtype=x.dtype)


def indi_int(params: IndiParams, x: NDArray, order: int) -> NDArray:
    # lower and upper bounds
    lb, ub = params

    val = np.zeros(x.size, dtype=x.dtype)
    ind0 = (x >= lb[0]) & (x <= ub[0])
    ind1 = x > ub[0]
    val[ind0] = (x[ind0] - lb[0])**(-order)/factorial(-order)
    for i in range(-order):
        val[ind1] += (
            ((ub[0] - lb[0])**(-order - i)/factorial(-order - i)) *
            ((x[ind1] - ub[0])**i/factorial(i))
        )
    return val


class Indi(BundleXFunction):

    def __init__(self, params: IndiParams) -> None:
        super().__init__(params, indi_val, indi_der, indi_int)
