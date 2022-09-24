import numpy as np

from xspline.indi import indi_int
from xspline.typing import (BsplParams, NDArray, NegativeInt, PositiveInt,
                            RawFunction)
from xspline.xfunction import BundleXFunction


def cache_bspl(function: RawFunction) -> RawFunction:
    cache = {}

    def wrapper_function(*args, **kwargs) -> NDArray:
        key = tuple(
            map(lambda x: id(x) if isinstance(x, np.ndarray) else x, args)
        )
        if key in cache:
            return cache[key]
        result = function(*args, **kwargs)
        cache[key] = result
        return result

    def cache_clear():
        cache.clear()

    wrapper_function.cache = cache
    wrapper_function.cache_clear = cache_clear
    return wrapper_function


@cache_bspl
def bspl_val(params: BsplParams, x: NDArray) -> NDArray:
    # knots, degree, and index
    t, k, i = params

    if k == 0:
        val = np.zeros(x.shape, dtype=x.dtype)
        val[(x >= t[i]) & (x < t[i + 1])] = 1.0
        if i == len(t) - 2:
            val[x == t[i + 1]] = 1.0
    else:
        ii = np.maximum(np.minimum([i, i + 1, i + k, i + k + 1], len(t) - 1), 0)

        val0 = np.zeros(x.shape, dtype=x.dtype)
        val1 = np.zeros(x.shape, dtype=x.dtype)

        if t[ii[0]] != t[ii[2]]:
            n0 = bspl_val((t, k - 1, i), x)
            val0 = (x - t[ii[0]])*n0/(t[ii[2]] - t[ii[0]])
        if t[ii[1]] != t[ii[3]]:
            n1 = bspl_val((t, k - 1, i + 1), x)
            val1 = (t[ii[3]] - x)*n1/(t[ii[3]] - t[ii[1]])

        val = val0 + val1

    return val


@cache_bspl
def bspl_der(params: BsplParams, x: NDArray, order: PositiveInt) -> NDArray:
    # knots, degree, and index
    t, k, i = params

    if order == 0:
        return bspl_val(params, x)

    if order > k:
        return np.zeros(x.shape, dtype=x.dtype)

    ii = np.maximum(np.minimum([i, i + 1, i + k, i + k + 1], len(t) - 1), 0)

    val0 = np.zeros(x.shape, dtype=x.dtype)
    val1 = np.zeros(x.shape, dtype=x.dtype)

    if t[ii[0]] != t[ii[2]]:
        n0 = bspl_der((t, k - 1, i), x, order - 1)
        val0 = k*n0/(t[ii[2]] - t[ii[0]])
    if t[ii[1]] != t[ii[3]]:
        n1 = bspl_der((t, k - 1, i + 1), x, order - 1)
        val1 = k*n1/(t[ii[3]] - t[ii[1]])

    val = val0 - val1

    return val


@cache_bspl
def bspl_int(params: BsplParams, x: NDArray, order: NegativeInt) -> NDArray:
    # knots, degree, and index
    t, k, i = params

    if order == 0:
        return bspl_val(params, x)

    ii = np.maximum(np.minimum([i, i + 1, i + k, i + k + 1], len(t) - 1), 0)
    if k == 0:
        val = np.zeros(x.shape, dtype=x.dtype)
        if t[ii[0]] != t[ii[1]]:
            val = indi_int(((t[ii[0]], True), (t[ii[1]], True)), x, order)
    else:
        val0 = np.zeros(x.shape, dtype=x.dtype)
        val1 = np.zeros(x.shape, dtype=x.dtype)

        if t[ii[0]] != t[ii[2]]:
            val0 = (
                (x - t[ii[0]])*bspl_int((t, k - 1, i), x, order) +
                order*bspl_int((t, k - 1, i), x, order - 1)
            )/(t[ii[2]] - t[ii[0]])
        if t[ii[1]] != t[ii[3]]:
            val1 = (
                (t[ii[3]] - x)*bspl_int((t, k - 1, i + 1), x, order) -
                order*bspl_int((t, k - 1, i + 1), x, order - 1)
            )/(t[ii[3]] - t[ii[1]])

        val = val0 + val1

    return val


class Bspl(BundleXFunction):

    def __init__(self, params: BsplParams) -> None:
        super().__init__(params, bspl_val, bspl_der, bspl_int)
        # partial strip the attributes, need to relink them
        # need a better solution
        self.val_fun.cache = bspl_val.cache
        self.val_fun.cache_clear = bspl_val.cache_clear
        self.der_fun.cache = bspl_der.cache
        self.der_fun.cache_clear = bspl_der.cache_clear
        self.int_fun.cache = bspl_int.cache
        self.int_fun.cache_clear = bspl_int.cache_clear

    def cache_clear(self) -> None:
        self.val_fun.cache_clear()
        self.der_fun.cache_clear()
        self.int_fun.cache_clear()
