from math import factorial

import numpy as np

from xspline.typing import PolyParams, NDArray
from xspline.xfunction import BundleXFunction, XFunction


def poly_val(params: PolyParams, x: NDArray) -> NDArray:
    """Polynomial value function.

    Parameters
    ----------
    params
        Polynomial coefficients.
    x
        Data points.

    Returns
    -------
    describe
        Polynomial function values.

    """
    return np.polyval(params, x)


def poly_der(params: PolyParams, x: NDArray, order: int) -> NDArray:
    """Polynomial derivative function.

    Parameters
    ----------
    params
        Polynomial coefficients.
    x
        Data points.
    order
        Order of differentiation.

    Returns
    -------
    describe
        Polynomial derivative values.

    """
    return np.polyval(np.polyder(params, order), x)


def poly_int(params: PolyParams, x: NDArray, order: int) -> NDArray:
    """Polynomial definite integral function.

    Parameters
    ----------
    params
        Polynomial coefficients.
    x
        Data points.
    order
        Order of integration. Here we use negative integer.

    Returns
    -------
    describe
        Polynomial definite integral values.

    """

    return np.polyval(np.polyint(params, -order), x)


class Poly(BundleXFunction):
    """Polynomial function. A simple wrapper for the numpy poly functions.

    Parameters
    ----------
    params
        This a tuple contains coefficients for the terms in polynomial.

    Example
    -------
    >>> poly = Poly((1.0, 0.0))
    >>> poly([0.0, 1.0])
    array([0.0, 1.0])
    >>> poly([0.0, 1.0], order=1)
    array([1.0, 1.0])
    >>> poly([0.0, 1.0], order=2)
    array([0.0, 0.0])
    >>> poly([0.0, 1.0]], order=-1)
    array([0.0, 0.5])

    """

    def __init__(self, params: PolyParams) -> None:
        super().__init__(params, poly_val, poly_der, poly_int)


def get_poly_params(fun: XFunction, x: float, degree: int) -> tuple[float, ...]:
    """Solve polynomial (taylor) coefficients provided the ``XFunction``.

    Parameters
    ----------
    fun
        Provided ``XFunction`` to be approximated.
    x
        The point where we want to approximate ``XFunction`` by the polynomial.
    degree
        Degree of the approximation polynomial.

    Returns
    -------
    describe
        The approximation polynomial coefficients.

    """
    if degree < 0:
        return (0.0,)
    rhs = np.array([fun(x, order=i) for i in range(degree, -1, -1)])
    mat = np.zeros((degree + 1, degree + 1))
    for i in range(degree + 1):
        for j in range(i + 1):
            mat[i, j] = factorial(degree - j) / factorial(i - j) * x ** (i - j)
    return tuple(np.linalg.solve(mat, rhs))


def get_poly_fun(fun: XFunction, x: float, degree: int) -> Poly:
    """Get the approximation polynomial function.

    Parameters
    ----------
    fun
        Provided ``XFunction`` to be approximated.
    x
        The point where we want to approximate ``XFunction`` by the polynomial.
    degree
        Degree of the approximation polynomial.

    Returns
    -------
    describe
        Instance of the ``Poly`` class to approximate provided ``XFunction``.

    """
    params = get_poly_params(fun, x, degree)
    return Poly(params)
