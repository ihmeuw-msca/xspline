import numpy as np

from xspline.typing import NDArray, PolyParams
from xspline.xfunction import BundleXFunction


def poly_val(params: PolyParams, x: NDArray) -> NDArray:
    return np.polyval(params, x)


def poly_der(params: PolyParams, x: NDArray, order: int) -> NDArray:
    return np.polyval(np.polyder(params, order), x)


def poly_int(params: PolyParams, x: NDArray, order: int) -> NDArray:
    return np.polyval(np.polyint(params, -order), x)


class Poly(BundleXFunction):

    def __init__(self, params: PolyParams) -> None:
        super().__init__(params, poly_val, poly_der, poly_int)
