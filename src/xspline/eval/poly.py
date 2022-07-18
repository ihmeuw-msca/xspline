import numpy as np
from numpy.typing import NDArray
from xspline.eval.typing import EvalFunction


class PolyEval(EvalFunction):

    @staticmethod
    def get_val(params: tuple, x: NDArray) -> NDArray:
        return np.polyval(params, x)

    @staticmethod
    def get_der(params: tuple, x: NDArray, p: int) -> NDArray:
        return np.polyval(np.polyder(params, p), x)

    @staticmethod
    def get_int(params: tuple, x: NDArray, p: int) -> NDArray:
        return np.polyval(np.polyint(params, -p), x)
