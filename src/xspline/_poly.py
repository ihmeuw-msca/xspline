import numpy as np
from numpy.typing import NDArray


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
