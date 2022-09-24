from abc import ABC, abstractmethod
from functools import partial
from math import factorial

from numpy.typing import NDArray

from xspline.typing import Optional, RawDFunction, RawIFunction, RawVFunction


class XFunction(ABC):

    @abstractmethod
    def __call__(self, x: NDArray,
                 order: int = 0,
                 start: Optional[NDArray] = None) -> NDArray:
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
            Raised when `order < 0` and `start = None`. Please proivde the
            starting points to compute definite integrals.

        """


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

    def __call__(self, x: NDArray,
                 order: int = 0,
                 start: Optional[NDArray] = None) -> NDArray:
        if order == 0:
            return self.val_fun(x)
        if order > 0:
            return self.der_fun(x, order)
        if start is None:
            raise ValueError("please provide starting points to compute definite integrals")
        dx = x - start
        val = self.int_fun(x, order)
        for i in range(-order):
            val -= self.int_fun(start, order + i)*(dx**i/factorial(i))
        return val
