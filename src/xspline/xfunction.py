from abc import ABC, abstractmethod
from functools import partial
from math import factorial

import numpy as np

from xspline.typing import (BoundaryPoint, NDArray, RawDFunction, RawIFunction,
                            RawVFunction)


def taylor_term(x: NDArray, order: int) -> NDArray:
    if x.ndim == 2:
        x = np.diff(x, axis=0)
    return x**order/factorial(order)


class XFunction(ABC):

    @abstractmethod
    def __call__(self, x: NDArray, order: int = 0) -> NDArray:
        """Function returns function values, derivatives and definite integrals.

        Parameters
        ----------
        x
            Data points where the function is evaluated.
        order
            Order of differentiation or integration. When `order = 0`, function
            value will be returned. When `order > 0` function derviative will
            be returned. When `order < 0`, `start` argument is requried and
            function integral will be returned. Default is `0`.
        start
            Starting points for the definite integral. This is required when
            `order < 0`. Default is `None`.

        Returns
        -------
        describe
            Return function values, derivatives or definite integrals.

        Raises
        ------
        ValueError
            Raised when `order >= 0` and `x` is not a 1d array. Please proivde a
            1d array when compute function values and derivatives.
        ValueError
            Raised when `order < 0` and `x` is not a 2d array with two rows.
            Please provide a 2d array with two rows when compute function
            definite integrals.

        """
        if order >= 0:
            if x.ndim != 1:
                raise ValueError("please provide a 1d array when compute "
                                 "function values and derivatives")
        if order < 0:
            if x.ndim != 2 or len(x) != 2:
                raise ValueError("please provide a 2d array with two rows when "
                                 "compute function definite integrals")

        if x.size == 0:
            return np.array([], dtype=x.dtype)

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

    def __call__(self, x: NDArray, order: int = 0) -> NDArray:
        super().__call__(x, order=order)
        if order == 0:
            return self.val_fun(x)
        if order > 0:
            return self.der_fun(x, order)
        dx = np.diff(x, axis=0)
        val = self.int_fun(x[1], order)
        for i in range(-order):
            val -= self.int_fun(x[0], order + i)*taylor_term(dx, i)
        return val
