from math import factorial

import numpy as np
import pytest
from numpy.typing import NDArray
from xspline.xfunction import XFunction

# test append function
sep = (1.0, False)


@XFunction
def fun0(x: NDArray, order: int = 0) -> NDArray:
    return np.zeros(x.shape[-1], dtype=x.dtype)


@XFunction
def fun1(x: NDArray, order: int = 0) -> NDArray:
    if order == 0:
        return np.ones(x.size, dtype=x.dtype)
    if order > 0:
        return np.zeros(x.size, dtype=x.dtype)
    dx = x[1] - x[0]
    return dx**(-order) / factorial(-order)


my_fun = fun0.append(fun1, sep)


def tr_fun(x: NDArray, order: int = 0) -> NDArray:
    left = x <= sep[0] if sep[1] else x < sep[0]
    if order == 0:
        return np.where(left, 0.0, 1.0)
    if order > 0:
        return np.zeros(x.size, dtype=x.dtype)
    z = np.maximum(0.0, x - sep[0])
    return z**(-order) / factorial(-order)


@pytest.mark.parametrize("order", [-2, -1, 0, 1, 2])
def test_append(order):
    x = np.linspace(0.0, 2.0, 101)
    my_val = my_fun(x, order=order)
    tr_val = tr_fun(x, order=order)
    assert np.allclose(my_val, tr_val)
