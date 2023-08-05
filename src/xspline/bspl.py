import numpy as np
from numpy.typing import NDArray

from xspline.indi import indi_int
from xspline.typing import BsplParams, RawFunction
from xspline.xfunction import BundleXFunction


def cache_bspl(function: RawFunction) -> RawFunction:
    """Cache implementation for bspline basis functions, to avoid repetitively
    evaluate functions.

    Parameters
    ----------
    function
        Raw value, derivative and definite integral functions.

    Returns
    -------
    describe
        Cached version of the raw functions.

    """
    cache = {}

    def wrapper_function(*args, **kwargs) -> NDArray:
        key = tuple(tuple(x.ravel()) if isinstance(x, np.ndarray) else x for x in args)
        if key in cache:
            return cache[key]
        result = function(*args, **kwargs)
        cache[key] = result
        return result

    def cache_clear():
        cache.clear()

    wrapper_function.cache_clear = cache_clear
    return wrapper_function


@cache_bspl
def bspl_val(params: BsplParams, x: NDArray) -> NDArray:
    """Value of the bspline function.

    Parameters
    ----------
    params
        Bspline function parameters as a tuple including, knots, degree and the
        index of the spline basis.
    x
        Data points.

    Returns
    -------
    describe
        Function value of the bspline function.

    """
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
            val0 = (x - t[ii[0]]) * n0 / (t[ii[2]] - t[ii[0]])
        if t[ii[1]] != t[ii[3]]:
            n1 = bspl_val((t, k - 1, i + 1), x)
            val1 = (t[ii[3]] - x) * n1 / (t[ii[3]] - t[ii[1]])

        val = val0 + val1

    return val


@cache_bspl
def bspl_der(params: BsplParams, x: NDArray, order: int) -> NDArray:
    """Derivative of the bspline function.

    Parameters
    ----------
    params
        Bspline function parameters as a tuple including, knots, degree and the
        index of the spline basis.
    x
        Data points.

    Returns
    -------
    describe
        Derivative of the bspline function.

    """
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
        val0 = k * n0 / (t[ii[2]] - t[ii[0]])
    if t[ii[1]] != t[ii[3]]:
        n1 = bspl_der((t, k - 1, i + 1), x, order - 1)
        val1 = k * n1 / (t[ii[3]] - t[ii[1]])

    val = val0 - val1

    return val


@cache_bspl
def bspl_int(params: BsplParams, x: NDArray, order: int) -> NDArray:
    """Definite integral of the bspline function.

    Parameters
    ----------
    params
        Bspline function parameters as a tuple including, knots, degree and the
        index of the spline basis.
    x
        Data points.

    Returns
    -------
    describe
        Definite integral of the bspline function.

    """
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
                (x - t[ii[0]]) * bspl_int((t, k - 1, i), x, order)
                + order * bspl_int((t, k - 1, i), x, order - 1)
            ) / (t[ii[2]] - t[ii[0]])
        if t[ii[1]] != t[ii[3]]:
            val1 = (
                (t[ii[3]] - x) * bspl_int((t, k - 1, i + 1), x, order)
                - order * bspl_int((t, k - 1, i + 1), x, order - 1)
            ) / (t[ii[3]] - t[ii[1]])

        val = val0 + val1

    return val


def clear_bspl_cache() -> None:
    """Clear all cache of the value, derivative and definite integral for
    bspline function.

    """
    bspl_val.cache_clear()
    bspl_der.cache_clear()
    bspl_int.cache_clear()


class Bspl(BundleXFunction):
    """Basis spline function.

    Parameters
    ----------
    params
        This is a tuple that contains knots, degree and index of the basis
        function.

    Example
    -------
    >>> bspl = Bspl(((0.0, 1.0), 1, 0)) # knots=(0.0, 1.0), degree=2, index=0
    >>> bspl([0.0, 1.0])
    array([0., 1.])
    >>> bspl([0.0, 1.0], order=1)
    array([1., 1.])
    >>> bspl([0.0, 1.0], order=2)
    array([0., 0.])
    >>> bspl([0.0, 1.0], order=-1)
    array([0. , 0.5])

    """

    def __init__(self, params: BsplParams) -> None:
        super().__init__(params, bspl_val, bspl_der, bspl_int)


def get_bspl_funs(knots: tuple[float, ...], degree: int) -> tuple[Bspl]:
    """Create the bspline basis functions give knots and degree.

    Parameters
    ----------
    knots
        Bspline knots.
    degree
        Bspline degree.

    Returns
    -------
    describe
        A full set of bspline functions.

    """
    return tuple(Bspl((knots, degree, i)) for i in range(-degree, len(knots) - 1))
