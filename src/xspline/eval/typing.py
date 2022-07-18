from math import factorial

import numpy as np
from numpy.typing import NDArray


class EvalFunction:
    """This class defines the interface of the evaluation function. The instance
    of this class is a callable object with evaluation function signature. To
    subclass this class, please provide the function `get_val`, `get_der` and
    `get_int` represents, value, dervative and integral function. The class
    will combine these three functions into a single evaluation function.

    Note
    ----
    When implementing `get_int` function, you don't need to specify the starting
    points of the integral. For example, if we use `np.polyint` for the
    implementation of the polynomial, the function always use 0 as the starting
    point. As long as the starting points are consistent across all order
    of integration, the final evaluation function should behave correctly.
    We use the following formula for the definte integral.

    .. math::

        I^k[f](a, b) = I^k[f](a_0, b) -
        \\sum_{i=1}^k \\frac{(b - a_0)^(k - i)}{(k - i)!} I^i[f](a_0, a)

    Here :math:`a_0` is the default starting point.

    Note
    ----
    When `p < 0` and `x` is a one dimentional array it will use the default
    starting point for the definite integral. The default starting point will
    depend on the actual implementation. It is really recommended to provide `x`
    as a two dimentional array that contains the starting and the end points of 
    the definite integrals, to avoid unexpected bevavior.

    """

    @staticmethod
    def get_val(params: tuple, x: NDArray) -> NDArray:
        raise NotImplementedError

    @staticmethod
    def get_der(params: tuple, x: NDArray, p: int) -> NDArray:
        raise NotImplementedError

    @staticmethod
    def get_int(params: tuple, x: NDArray, p: int) -> NDArray:
        raise NotImplementedError

    def __call__(self, params: tuple, x: NDArray, p: int = 0) -> NDArray:
        """The evaluation function.

        Parameters
        ----------
        parmas
            The parameter of the function.
        x
            Points where the function will be evaluated.
        p
            The order of differentiation or integration.

        Returns
        -------
        NDArray
            When `p == 0`, it will return the function value. When `p > 0`, it
            will return the derivatives. And when `p < 0`, it will return the
            integrals.

        """
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
