from typing import Optional

import numpy as np

from xspline.bundle import BundleXFunction
from xspline.indi import indi_int
from xspline.typing import BsplParams, NDArray, NegativeInt, PositiveInt


def bspl_val(params: BsplParams,
             x: NDArray,
             cache: Optional[dict] = None) -> NDArray:
    # knots, degree, and index
    t, k, i = params
    cache_key = (t, k, i, 0, id(x))
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    if k == 0:
        val = np.zeros(x.shape, dtype=x.dtype)
        val[(x >= t[i]) & (x < t[i + 1])] = 1.0
        if i == t.size - 2:
            val[x == t[i + 1]] = 1.0
    else:
        ii = np.maximum(np.minimum([i, i + 1, i + k, i + k + 1], t.size - 1), 0)

        val0 = np.zeros(x.shape, dtype=x.dtype)
        val1 = np.zeros(x.shape, dtype=x.dtype)

        if t[ii[0]] != t[ii[2]]:
            n0 = bspl_val((t, k - 1, i), x, cache=cache)
            val0 = (x - t[ii[0]])*n0/(t[ii[2]] - t[ii[0]])
        if t[ii[1]] != t[ii[3]]:
            n1 = bspl_val((t, k - 1, i + 1), x, cache=cache)
            val1 = (t[ii[3]] - x)*n1/(t[ii[3]] - t[ii[1]])

        val = val0 + val1

    if cache is not None:
        cache[cache_key] = val
    return val


def bspl_der(params: BsplParams,
             x: NDArray,
             order: PositiveInt,
             cache: Optional[dict] = None) -> NDArray:
    # knots, degree, and index
    t, k, i = params
    cache_key = (t, k, i, order, id(x))
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    if order > k:
        return np.zeros(x.shape, dtype=x.dtype)

    ii = np.maximum(np.minimum([i, i + 1, i + k, i + k + 1], t.size - 1), 0)

    val0 = np.zeros(x.shape, dtype=x.dtype)
    val1 = np.zeros(x.shape, dtype=x.dtype)

    if t[ii[0]] != t[ii[2]]:
        n0 = bspl_der((t, k - 1, i), x, order - 1, cache=cache)
        val0 = k*n0/(t[ii[2]] - t[ii[0]])
    if t[ii[1]] != t[ii[3]]:
        n1 = bspl_der((t, k - 1, i + 1), x, order - 1, cache=cache)
        val1 = k*n1/(t[ii[3]] - t[ii[1]])

    val = val0 - val1

    if cache is not None:
        cache[cache_key] = val
    return val


def bspl_int(params: BsplParams,
             x: NDArray,
             order: NegativeInt,
             cache: Optional[dict] = None) -> NDArray:
    # knots, degree, and index
    t, k, i = params
    cache_key = (t, k, i, order, id(x))
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    ii = np.maximum(np.minimum([i, i + 1, i + k, i + k + 1], t.size - 1), 0)
    if k == 0:
        val = np.zeros(x.shape, dtype=x.dtype)
        if t[ii[0]] != t[ii[1]]:
            val = indi_int(((t[ii[0]], True), (t[ii[1]], True)), x, order)
    else:
        val0 = np.zeros(x.shape, dtype=x.dtype)
        val1 = np.zeros(x.shape, dtype=x.dtype)

        if t[ii[0]] != t[ii[2]]:
            val0 = (
                (x - t[ii[0]])*bspl_int((t, k - 1, i), x, order, cache=cache) +
                order*bspl_int((t, k - 1, i), x, order - 1, cache=cache)
            )/(t[ii[2]] - t[ii[0]])
        if t[ii[1]] != t[ii[3]]:
            val1 = (
                (t[ii[3]] - x)*bspl_int((t, k - 1, i + 1), x, order, cache=cache) -
                order*bspl_int((t, k - 1, i + 1), x, order - 1, cache=cache)
            )/(t[ii[3]] - t[ii[1]])

        val = val0 + val1

    if cache is not None:
        cache[cache_key] = val
    return val


class Bspl(BundleXFunction):

    def __init__(self, params: BsplParams) -> None:
        super().__init__(params, bspl_val, bspl_der, bspl_int)
