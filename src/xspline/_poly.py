import numpy as np
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
