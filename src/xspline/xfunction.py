from __future__ import annotations
from functools import partial
from math import factorial
from operator import attrgetter

import numpy as np

from xspline.typing import (
    BoundaryPoint,
    Callable,
    RawDFunction,
    RawIFunction,
    RawVFunction,
    NDArray,
)


class XFunction:
    """Function interface that provide easy access to function value,
    derivatives and definite integrals.

    Parameters
    ----------
    fun
        Function implementation.

    TODO: describe the interface of the function implementation

    """

    def __init__(self, fun: Callable) -> None:
        self._fun = fun

    def _check_args(self, x: NDArray, order: int) -> tuple[NDArray, int, bool]:
        x, order = np.asarray(x, dtype=float), int(order)
        if (x.ndim not in [0, 1, 2]) or (x.ndim == 2 and len(x) != 2):
            raise ValueError(
                "please provide a scalar, an 1d array, or a 2d " "array with two rows"
            )

        # reshape array
        isscalar = x.ndim == 0
        if isscalar:
            x = x.ravel()
        if order >= 0 and x.ndim == 2:
            raise ValueError(
                "please provide an 1d array for function value "
                "defivative computation"
            )
        if order < 0 and x.ndim == 1:
            x = np.vstack([np.repeat(x.min(), x.size), x])

        # check interval bounds
        if order < 0 and (x[0] > x[1]).any():
            raise ValueError("to integrate, `x` must satisfy `x[0] <= x[1]`")

        return x, order, isscalar

    def __call__(self, x: NDArray, order: int = 0) -> NDArray:
        """Function returns function values, derivatives and definite integrals.

        Parameters
        ----------
        x
            Data points where the function is evaluated. If `order < 0` and `x`
            is a 2d array with two rows, the rows will be treated as the
            starting and ending points for definite interval. If `order < 0` and
            `x` is a 1d array, function will use the smallest number in `x` as
            the starting point of the definite interval.
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
        ValueErorr
            Raised when `order >= 0` and `x` is a 2d array.
        ValueError
            Raised when `order < 0` and `any(x[0] > x[1])`.

        """
        x, order, isscalar = self._check_args(x, order)
        if x.size == 0:
            return np.empty(shape=x.shape, dtype=x.dtype)
        result = self._fun(x, order)
        if isscalar:
            return result[0]
        return result

    def append(self, other: XFunction, sep: BoundaryPoint) -> XFunction:
        """Splice with another instance of ``XFunction`` to create a new
        ``XFunction``.

        Parameters
        ----------
        other
            Another ``XFunction`` after the current function.
        sep
            The boundary point to separate two functions, before is the current
            function and after the the ``other`` function.

        """
        lfun, rfun = self._fun, other._fun

        def fun(x: NDArray, order: int = 0) -> NDArray:
            left = x <= sep[0] if sep[1] else x < sep[0]

            if order >= 0:
                return np.where(left, lfun(x, order), rfun(x, order))

            lboth, rboth = left.all(axis=0), (~left).all(axis=0)
            landr = (~lboth) & (~rboth)

            result = np.zeros(x.shape[1], dtype=x.dtype)
            result[lboth] = lfun(x[:, lboth], order)
            result[rboth] = rfun(x[:, rboth], order)

            if landr.any():
                lx = np.insert(x[np.ix_([0], landr)], 1, sep[0], axis=0)
                rx = np.insert(x[np.ix_([1], landr)], 0, sep[0], axis=0)
                dx = x[1][landr] - sep[0]

                for i in range(1, -order):
                    result[landr] += lfun(lx, order + i) * (dx**i / factorial(i))
                result[landr] += lfun(lx, order) + rfun(rx, order)
            return result

        return XFunction(fun)


class BundleXFunction(XFunction):
    """This is one implementation of the ``XFunction``, it takes the value,
    derivative and definite integral function and bundle them together as a
    ``XFunction``.

    Parameters
    ----------
    params
        This is the parameters that is needed for the value, derivatives and
        the definitely integral function.
    val_fun
        Value function.
    der_fun
        Derviative function.
    int_fun
        Defintie integral function.

    """

    def __init__(
        self,
        params: tuple,
        val_fun: RawVFunction,
        der_fun: RawDFunction,
        int_fun: RawIFunction,
    ) -> None:
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
                val -= self.int_fun(x[0], order + i) * (dx**i / factorial(i))
            return val

        super().__init__(fun)


class BasisXFunction(XFunction):
    """This is one implementation of ``XFunction`` by taking in a set of
    instances of ``XFunction`` as basis functions. And the linear combination
    coefficients to provide function value, derivative and definite integral.

    Parameters
    ----------
    basis_funs
        A set of instances of ``XFunction`` as basis functions.
    coef
        Coefficients for the linearly combine the basis functions.

    """

    coef = property(attrgetter("_coef"))

    def __init__(
        self, basis_funs: tuple[XFunction, ...], coef: NDArray | None = None
    ) -> None:
        if not all(isinstance(fun, XFunction) for fun in basis_funs):
            raise TypeError("basis functions must all be instances of 'XFunction'")
        self.basis_funs = tuple(basis_funs)
        self.coef = coef

        def fun(x: NDArray, order: int = 0) -> NDArray:
            if self.coef is None:
                raise ValueError(
                    "please provide the coefficients for the basis functions"
                )
            design_mat = self.get_design_mat(x, order=order, check_args=False)
            return design_mat.dot(self.coef)

        super().__init__(fun)

    @coef.setter
    def coef(self, coef: NDArray | None) -> None:
        if coef is not None:
            coef = np.asarray(coef, dtype=float).ravel()
            if coef.size != len(self):
                raise ValueError(
                    "number of coeffcients does not match number of basis functions"
                )
        self._coef = coef

    def get_design_mat(
        self, x: NDArray, order: int = 0, check_args: bool = True
    ) -> NDArray:
        """Provide design matrix from the set of basis functions.

        Parameters
        ----------
        x
            Data points
        order
            Order of differentiation/integration.
        check_args
            If ``True`` it will check and parse the arguments.

        Returns
        -------
        describe
            Design matrix with dimention number of data points by number of
            basis functions.

        """
        if check_args:
            x, order, _ = self._check_args(x, order)
        return np.vstack([xfun._fun(x, order) for xfun in self.basis_funs]).T

    def __len__(self) -> int:
        """Number of basis functions."""
        return len(self.basis_funs)
