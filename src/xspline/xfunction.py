from functools import partial
from math import factorial
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from xspline.typing import (BoundaryPoint, Callable, RawDFunction,
                            RawIFunction, RawVFunction)


def taylor_term(x: NDArray, order: int) -> NDArray:
    if x.ndim == 2:
        x = np.diff(x, axis=0)[0]
    return x**order/factorial(order)


class XFunction:

    def __init__(self, fun: Optional[Callable] = None) -> None:
        if not hasattr(self, "fun"):
            self.fun = fun

    def __call__(self, x: NDArray, order: int = 0) -> NDArray:
        """Function returns function values, derivatives and definite integrals.

        Parameters
        ----------
        x
            Data points where the function is evaluated. If `order >= 0` and
            `x` is a 2d array with two rows, the difference between the rows
            will be used for function evaluation. If `order < 0` and `x` is a
            2d array with two rows, the rows will be treated as the starting and
            ending points for definite interval. If `order < 0` and `x` is a
            1d array, function will use the smallest number in `x` as the
            starting point of the definite interval.
        order
            Order of differentiation or integration. When `order = 0`, function
            value will be returned. When `order > 0` function derviative will
            be returned. When `order < 0`, function integral will be returned.
            Default is `0`.

        Returns
        -------
        describe
            Return function values, derivatives or definite integrals.

        Raises
        ------
        AttributeError
            Raised when the function implementation is not provided.
        ValueError
            Raised when `x` is not a scalar, 1d array or 2d array with two rows.

        """
        # validate
        if getattr(self, "fun", None) is None:
            raise AttributeError("please provide the function implementation")
        x = np.asarray(x, dtype=float)
        if (x.ndim not in [0, 1, 2]) or (x.ndim == 2 and len(x) != 2):
            raise ValueError("please provide a scalar, an 1d array, or a 2d "
                             "array with two rows")

        # special case, empty array
        if x.size == 0:
            return np.empty(shape=x.shape, dtype=x.dtype)

        # reshape array
        isscalar = x.ndim == 0
        if isscalar:
            x = x.ravel()
        if order >= 0 and x.ndim == 2:
            x = x[1] - x[0]
        if order < 0 and x.ndim == 1:
            x = np.vstack([np.repeat(x.min(), x.size), x])

        # call fun
        result = self.fun(x, order)

        # return scalar if input is scalar
        if isscalar:
            result = result[0]
        return result

    def add(self, other: "XFunction", sep: BoundaryPoint) -> "XFunction":

        def fun(x: NDArray, order: int = 0) -> NDArray:
            li = x <= sep[0] if sep[1] else x < sep[0]
            ri = ~li

            val = np.zeros(x.size, dtype=x.dtype)
            if order >= 0:
                val[li] = self(x[li], order)
                val[ri] = other(x[ri], order)
            else:
                val[li] = self(x[li], order)
                if ri.any():
                    lx = np.repeat(sep[0], ri.sum())
                    rx = np.vstack([lx, x[ri]])
                    for i in range(order + 1, 0):
                        val[ri] += self(lx, order)*taylor_term(rx, i - order)
                    val[ri] += self(lx, order) + other(rx, order)
            return val

        return fun


class BundleXFunction(XFunction):

    def __init__(self,
                 params: tuple,
                 val_fun: RawVFunction,
                 der_fun: RawDFunction,
                 int_fun: RawIFunction) -> None:
        self.params = params
        self.val_fun = partial(val_fun, params)
        self.der_fun = partial(der_fun, params)
        self.int_fun = partial(int_fun, params)

        def fun(x: NDArray, order: int = 0) -> NDArray:
            if order == 0:
                return self.val_fun(x)
            if order > 0:
                return self.der_fun(x, order)
            dx = np.diff(x, axis=0)[0]
            val = self.int_fun(x[1], order)
            for i in range(-order):
                val -= self.int_fun(x[0], order + i)*taylor_term(dx, i)
            return val

        super().__init__(fun=fun)
