from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .utils import indicator_if


def bspl_evl(x: NDArray,
             p: int,
             t: NDArray,
             k: int,
             i: int,
             use_cache: bool = True) -> NDArray:
    if x.ndim == 2:
        val0 = _bspl_evl(x[0], p, t, k, i, use_cache=use_cache)
        val1 = _bspl_evl(x[1], p, t, k, i, use_cache=use_cache)
        return val1 - val0
    return _bspl_evl(x, p, t, k, i, use_cache=use_cache)


def _bspl_evl(x: NDArray,
              p: int,
              t: NDArray,
              k: int,
              i: int,
              use_cache: bool = True) -> NDArray:
    cache = {} if use_cache else None
    if p < 0:
        return bspl_int(x, p, t, k, i, cache=cache)
    if p > 0:
        return bspl_der(x, p, t, k, i, cache=cache)
    return bspl_val(x, t, k, i, cache=cache)


def bspl_val(x: NDArray,
             t: NDArray,
             k: int,
             i: int,
             cache: Optional[dict] = None) -> NDArray:
    """Evaluate basis-spline functions.

    Parameters
    ----------
    x
        Points where the function is evaluated.
    t
        Knots of the spline, assume to be non-decreasing sequence.
    k
        Degree of the spline, assume to be non-negative.
    i
        Index of the basis spline function, assume to be between `-k` and
        `len(t) - 2`.
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

        val0 = np.zeros(x.shape, dtype=x.dtype)
        val1 = np.zeros(x.shape, dtype=x.dtype)

        if t[ii[0]] != t[ii[2]]:
            n0 = bspl_val(x, t, k - 1, i, cache=cache)
            val0 = (x - t[ii[0]])*n0/(t[ii[2]] - t[ii[0]])
        if t[ii[1]] != t[ii[3]]:
            n1 = bspl_val(x, t, k - 1, i + 1, cache=cache)
            val1 = (t[ii[3]] - x)*n1/(t[ii[3]] - t[ii[1]])

        val = val0 + val1

    if cache is not None:
        cache[(k, i, 0)] = val
    return val


def bspl_der(x: NDArray,
             p: int,
             t: NDArray,
             k: int,
             i: int,
             cache: Optional[dict] = None) -> NDArray:
    """Evaluate derivatives of basis-spline functions.

    Parameters
    ----------
    x
        Points where the function is evaluated.
    p
        Order of differentiation, assume to be non-negative. When `p=0`, it will
        return the basis spline function value from `bspl_val`.
    t
        Knots of the spline, assume to be non-decreasing sequence.
    k
        Degree of the spline, assume to be non-negative.
    i
        Index of the basis spline function, assume to be between `-k` and
        `len(t) - 2`.
    cache
        Optional cache dictionary to save computation. Default to be `None`.
        When `cache=None`, cache will not be used in the computation.

    Returns
    -------
    NDArray
        Spline derivative values evaluated at given points.

    """
    if (cache is not None) and ((k, i, p) in cache):
        return cache[(k, i, p)]

    if p == 0:
        return bspl_val(x, t, k, i, cache=cache)

    if p > k:
        return np.zeros(x.shape, dtype=x.dtype)

    ii = np.maximum(np.minimum([i, i + 1, i + k, i + k + 1], t.size - 1), 0)

    val0 = np.zeros(x.shape, dtype=x.dtype)
    val1 = np.zeros(x.shape, dtype=x.dtype)

    if t[ii[0]] != t[ii[2]]:
        n0 = bspl_der(x, p - 1, t, k - 1, i, cache=cache)
        val0 = k*n0/(t[ii[2]] - t[ii[0]])
    if t[ii[1]] != t[ii[3]]:
        n1 = bspl_der(x, p - 1, t, k - 1, i + 1, cache=cache)
        val1 = k*n1/(t[ii[3]] - t[ii[1]])

    val = val0 - val1

    if cache is not None:
        cache[(k, i, p)] = val
    return val


def bspl_int(x: NDArray,
             p: int,
             t: NDArray,
             k: int,
             i: int,
             cache: Optional[dict] = None) -> NDArray:
    """Evaluate integrals of basis-spline functions.

    Parameters
    ----------
    x
        Points where the function is evaluated.
    p
        Order of integration, assume to be non-positive (we use negative number
        to denote the order of integration to distinguish with differentiation).
        When `p=0`, it will return the basis spline function value from
        `bspl_val`.
    t
        Knots of the spline, assume to be non-decreasing sequence.
    k
        Degree of the spline, assume to be non-negative.
    i
        Index of the basis spline function, assume to be between `-k` and
        `len(t) - 2`.
    cache
        Optional cache dictionary to save computation. Default to be `None`.
        When `cache=None`, cache will not be used in the computation.

    Returns
    -------
    NDArray
        Spline integral values evaluated at given points.

    """
    if (cache is not None) and ((k, i, p) in cache):
        return cache[(k, i, p)]

    if p == 0:
        return bspl_val(x, t, k, i, cache=cache)

    ii = np.maximum(np.minimum([i, i + 1, i + k, i + k + 1], t.size - 1), 0)
    if k == 0:
        val = np.zeros(x.shape, dtype=x.dtype)
        if t[ii[0]] != t[ii[1]]:
            indices = x > t[ii[0]]
            val[indices] = indicator_if(t[ii[0]], x[indices], -p,
                                        np.array([t[ii[0]], t[ii[1]]]))
    else:
        val0 = np.zeros(x.shape, dtype=x.dtype)
        val1 = np.zeros(x.shape, dtype=x.dtype)

        if t[ii[0]] != t[ii[2]]:
            val0 = (
                (x - t[ii[0]])*bspl_int(x, p, t, k - 1, i, cache=cache) +
                p*bspl_int(x, p - 1, t, k - 1, i, cache=cache)
            )/(t[ii[2]] - t[ii[0]])
        if t[ii[1]] != t[ii[3]]:
            val1 = (
                (t[ii[3]] - x)*bspl_int(x, p, t, k - 1, i + 1, cache=cache) -
                p*bspl_int(x, p - 1, t, k - 1, i + 1, cache=cache)
            )/(t[ii[3]] - t[ii[1]])

        val = val0 + val1

    if cache is not None:
        cache[(k, i, p)] = val
    return val
