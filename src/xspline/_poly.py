from math import comb

import numpy as np
from numpy.polynomial.polynomial import polyvander
from numpy.typing import NDArray


def poly_evl(x: NDArray,
             p: int,
             c: NDArray) -> NDArray:
    if x.ndim == 2:
        val0 = _poly_evl(x[0], p, c)
        val1 = _poly_evl(x[1], p, c)
        return val1 - val0
    return _poly_evl(x, p, c)


def _poly_evl(x: NDArray,
              p: int,
              c: NDArray) -> NDArray:
    if p < 0:
        return poly_int(x, p, c)
    if p > 0:
        return poly_der(x, p, c)
    return poly_val(x, c)


def poly_val(x: NDArray,
             c: NDArray) -> NDArray:
    return np.polyval(c, x)


def poly_der(x: NDArray,
             p: int,
             c: NDArray) -> NDArray:
    if p == 0:
        return poly_val(x, c)
    return np.polyval(np.polyder(c, p), x)


def poly_int(x: NDArray,
             p: int,
             c: NDArray) -> NDArray:
    if p == 0:
        return poly_val(x, c)
    return np.polyval(np.polyint(c, -p), x)


def shift_poly_coefs(c: NDArray, x: float) -> NDArray:
    n = len(c) - 1
    c_mat = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(i + 1):
            c_mat[i, j] = c[i - j]*comb(n - i + j, n - i)
    return c_mat.dot(polyvander(-x, n)[0])
