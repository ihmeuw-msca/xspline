from math import factorial

import numpy as np

from xspline.typing import IndiParams, NDArray
from xspline.xfunction import BundleXFunction


def indi_val(params: IndiParams, x: NDArray) -> NDArray:
    """Indicator value function,

    Parameters
    ----------
    params
        Indicator parameters as a tuple consists of lower and upper bound of
        the interval corresponding to the indicator function.
    x
        Data points.

    Returns
    -------
    describe
        Indicator function value.

    """
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
    """Indicator derivative function. Since indicator function is a piecewise
    constant function, its derivative will always be zero.

    Parameters
    ----------
    params
        Indicator parameters as a tuple consists of lower and upper bound of
        the interval corresponding to the indicator function.
    x
        Data points.

    Returns
    -------
    describe
        Indicator deviative value.

    """
    return np.zeros(x.size, dtype=x.dtype)


def indi_int(params: IndiParams, x: NDArray, order: int) -> NDArray:
    """Indicator definite integral function. It is a piecewise polynomial
    function.

    Parameters
    ----------
    params
        Indicator parameters as a tuple consists of lower and upper bound of
        the interval corresponding to the indicator function.
    x
        Data points.

    Returns
    -------
    describe
        Indicator definite integral value.

    """
    # lower and upper bounds
    lb, ub = params

    val = np.zeros(x.size, dtype=x.dtype)
    ind0 = (x >= lb[0]) & (x <= ub[0])
    ind1 = x > ub[0]
    val[ind0] = (x[ind0] - lb[0]) ** (-order) / factorial(-order)
    for i in range(-order):
        val[ind1] += ((ub[0] - lb[0]) ** (-order - i) / factorial(-order - i)) * (
            (x[ind1] - ub[0]) ** i / factorial(i)
        )
    return val


class Indi(BundleXFunction):
    """Indicator function.

    Parameters
    ----------
    params
        This is a tuple contains the lower and upper bounds of the indicator
        function. For each bound it consists of a number for the location of the
        bound and a boolean for the inclusion of the bound. For example, if we
        pass in `((0.0, True), (1.0, False))`, this represents interval [0, 1).

    Example
    -------
    >>> indi = Indi(((0.0, True), (1.0, False)))
    >>> indi([-1.0, 0.0, 1.0])
    array([0., 1., 0.])
    >>> indi([-1.0, 0.0, 1.0], order=1)
    array([0., 0., 0.])
    >>> indi([-1.0, 0.0, 1.0], order=-1)
    array([0., 0., 1.])

    """

    def __init__(self, params: IndiParams) -> None:
        super().__init__(params, indi_val, indi_der, indi_int)
