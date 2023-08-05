from math import factorial

import numpy as np
import pytest
from numpy.typing import NDArray
from xspline.xfunction import XFunction


@pytest.mark.parametrize("order", [-2, -1, 0, 1, 2])
def test_append(order):
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
        return dx ** (-order) / factorial(-order)

    my_fun = fun0.append(fun1, sep)

    def tr_fun(x: NDArray, order: int = 0) -> NDArray:
        left = x <= sep[0] if sep[1] else x < sep[0]
        if order == 0:
            return np.where(left, 0.0, 1.0)
        if order > 0:
            return np.zeros(x.size, dtype=x.dtype)
        z = np.maximum(0.0, x - sep[0])
        return z ** (-order) / factorial(-order)

    x = np.linspace(0.0, 2.0, 101)
    my_val = my_fun(x, order=order)
    tr_val = tr_fun(x, order=order)
    assert np.allclose(my_val, tr_val)


@pytest.fixture
def xfun():
    @XFunction
    def fun(x: NDArray, order: int = 0) -> NDArray:
        if order == 0:
            return np.ones(x.size, dtype=x.dtype)
        if order > 0:
            return np.zeros(x.size, dtype=x.dtype)
        dx = x[1] - x[0]
        return dx ** (-order) / factorial(-order)

    return fun


def test_check_args_0(xfun):
    """check value error, when x dimension is greater than 2"""
    x = np.ones((2, 2, 2))
    with pytest.raises(ValueError):
        xfun(x)


def test_check_args_1(xfun):
    """check value error, when x dimension is 2 but x have more rows than 2"""
    x = np.ones((3, 2))
    with pytest.raises(ValueError):
        xfun(x)


@pytest.mark.parametrize("order", [0, 1])
def test_check_args_2(xfun, order):
    """check collapes behavior when x dimension is 2, and compute val or der"""
    x = np.ones((2, 3))
    with pytest.raises(ValueError):
        xfun._check_args(x, order=order)


def test_check_args_3(xfun):
    """check extend behavior when x dimension is 1, and compute integral"""
    x = np.ones(3)
    x, _, _ = xfun._check_args(x, order=-1)
    assert np.allclose(x, np.ones((2, 3)))


def test_check_args_4(xfun):
    """check when x is a scalar"""
    x = 1.0
    x, _, isscalar = xfun._check_args(x, 0)
    assert isscalar and x.shape == (1,) and np.allclose(x, 1.0)


def test_scalar_input_output(xfun):
    """check when input is a scalar, output is also a scalar"""
    x = 1.0
    y = xfun(x)
    assert np.isscalar(y)


def test_empty_input_output(xfun):
    """check when input is an empty array, output is also an empty array."""
    x = np.array([], dtype=float)
    y = xfun(x)
    assert y.size == 0 and y.shape == (0,)
