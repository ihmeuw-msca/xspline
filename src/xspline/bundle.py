from functools import partial
from math import factorial
from typing import Optional

from numpy.typing import NDArray

from xspline.typing import RawDFunction, RawIFunction, RawVFunction


class BundleXFunction:

    def __init__(self,
                 params: tuple,
                 val_fun: RawVFunction,
                 der_fun: RawDFunction,
                 int_fun: RawIFunction) -> None:
        self.params = params
        self.val_fun = partial(val_fun, params)
        self.der_fun = partial(der_fun, params)
        self.int_fun = partial(int_fun, params)

    def __call__(self,
                 x: NDArray,
                 order: int = 0,
                 start: Optional[NDArray] = None,
                 **kwargs) -> NDArray:
        if order == 0:
            return self.val_fun(x, **kwargs)
        if order > 0:
            return self.der_fun(x, order, **kwargs)
        if start is None:
            raise ValueError("please provide starting points for computing definite intergral")
        dx = x - start
        val = self.int_fun(x, order, **kwargs)
        for i in range(-order):
            val -= self.int_fun(start, order + i, **kwargs)*(dx**i/factorial(i))
        return val
