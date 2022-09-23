import numpy as np

from xspline.bundle import BundleXFunction
from xspline.typing import NDArray, NegativeInt, PolyParams, PositiveInt


def poly_val(params: PolyParams, x: NDArray) -> NDArray:
    return np.polyval(params, x)


def poly_der(params: PolyParams, x: NDArray, order: PositiveInt) -> NDArray:
    return np.polyval(np.polyder(params, order), x)


def poly_int(params: PolyParams, x: NDArray, order: NegativeInt) -> NDArray:
    return np.polyval(np.polyint(params, -order), x)


class Poly(BundleXFunction):

    def __init__(self, params: PolyParams) -> None:
        super().__init__(params, poly_val, poly_der, poly_int)
