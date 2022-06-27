from typing import Optional

import numpy as np
from numpy.typing import NDArray


def spl_evl(t: NDArray,
            k: int,
            i: int,
            x: NDArray,
            cache: Optional[dict] = None) -> NDArray:
    """Evaluate basis spline functions.

    Parameters
    ----------
    t
        Knots of the spline, assume to be non-decreasing sequence.
    k
        Degree of the spline, assume to be non-negative.
    i
        Index of the basis spline function, assume to be between `-k` and
        `len(t) - 2`.
    x
        Points where the function is evaluated.
    cache
        Optional cache dictionary to save computation. Default to be `None`.
        When `cache=None`, cache will not be used in the computation.

    Returns
    -------
    NDArray
        Spline values evaluated at given points.

    """
    if (cache is not None) and ((k, i, 0) in cache):
        return cache[(k, i, 0)]

    if k == 0:
        val = np.zeros(x.shape, dtype=x.dtype)
        val[(x >= t[i]) & (x < t[i + 1])] = 1.0
        if i == t.size - 2:
            val[x == t[i + 1]] = 1.0
    else:
        ii = np.maximum(np.minimum([i, i + 1, i + k, i + k + 1], t.size - 1), 0)

        val0 = 0.0
        val1 = 0.0

        if ii[0] != ii[2]:
            n0 = spl_evl(t, k - 1, i, x, cache=cache)
            val0 = (x - t[ii[0]])*n0/(t[ii[2]] - t[ii[0]])
        if ii[1] != ii[3]:
            n1 = spl_evl(t, k - 1, i + 1, x, cache=cache)
            val1 = (t[ii[3]] - x)*n1/(t[ii[3]] - t[ii[1]])

        val = val0 + val1

    if cache is not None:
        cache[(k, i, 0)] = val
    return val
