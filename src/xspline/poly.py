from math import factorial

import numpy as np
from numpy.typing import NDArray

from xspline.typing import PolyParams
from xspline.xfunction import BundleXFunction, XFunction


def poly_val(params: PolyParams, x: NDArray) -> NDArray:
    return np.polyval(params, x)


def poly_der(params: PolyParams, x: NDArray, order: int) -> NDArray:
    return np.polyval(np.polyder(params, order), x)


def poly_int(params: PolyParams, x: NDArray, order: int) -> NDArray:
    return np.polyval(np.polyint(params, -order), x)


class Poly(BundleXFunction):

    def __init__(self, params: PolyParams) -> None:
        super().__init__(params, poly_val, poly_der, poly_int)


def get_poly_params(fun: XFunction, x: float, degree: int) -> tuple[float, ...]:
    if degree < 0:
        return (0.0,)
    rhs = np.array([fun(x, order=i) for i in range(degree, -1, -1)])
    mat = np.zeros((degree + 1, degree + 1))
    for i in range(degree + 1):
        for j in range(i + 1):
            mat[i, j] = factorial(degree - j) / factorial(i - j) * x**(i - j)
    return tuple(np.linalg.solve(mat, rhs))


def get_poly_fun(fun: XFunction, x: float, degree: int) -> Poly:
    params = get_poly_params(fun, x, degree)
    return Poly(params)
