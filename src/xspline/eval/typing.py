from math import factorial

import numpy as np
from numpy.typing import NDArray


class EvalFunction:

    @staticmethod
    def get_val(params: tuple, x: NDArray) -> NDArray:
        raise NotImplementedError

    @staticmethod
    def get_der(params: tuple, x: NDArray, p: int) -> NDArray:
        raise NotImplementedError

    @staticmethod
    def get_int(params: tuple, x: NDArray, p: int) -> NDArray:
        raise NotImplementedError

    def __call__(self, params: tuple, x: NDArray, p: int) -> NDArray:
        if x.size == 0:
            return np.array([], dtype=x.dtype)
        if p >= 0 and x.ndim == 2:
            x = x[-1]
        if p == 0:
            return self.get_val(params, x)
        if p > 0:
            return self.get_der(params, x, p)
        if x.ndim == 2:
            val = self.get_int(params, x[1], p)
            dx = x[1] - x[0]
            for i in range(-p):
                val -= self.get_int(params, x[0], p + i)*(dx**i/factorial(i))
            return val
        return self.get_int(params, x, p)
