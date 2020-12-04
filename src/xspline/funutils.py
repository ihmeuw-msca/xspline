"""
Function Utility Modules
"""
from collections.abc import Iterable
from numbers import Number
from typing import Tuple, Type

import numpy as np
from scipy.interpolate import lagrange

from xspline.interval import Interval


def check_number(num: Number, numtype: Type = None, invl: Interval = None):
    if numtype is not None:
        assert isinstance(num, Number)
        assert issubclass(numtype, Number)
        num = numtype(num)
    if invl is not None:
        assert num in invl
    return num


def check_fun_input(data: Iterable, order: Number) -> Tuple[np.ndarray, int]:
    data = np.asarray(data)
    data = data[None, :] if data.ndim == 1 else data
    order = check_number(order, int)
    assert data.ndim == 2
    assert len(data) in [1, 2]
    assert order >= 0 or len(data) == 2
    return data, order


def taylor_term(data: Iterable, order: Number) -> np.ndarray:
    data, order = check_fun_input(data, order)
    assert order >= 0
    data = data.ravel() if len(data) == 1 else data[1] - data[0]
    return data**order/np.math.factorial(order)


def lag_fun(data: Iterable, weights: Iterable, invl: Interval) -> np.ndarray:
    assert not np.isinf(invl.size)
    assert len(weights) >= 2
    data = np.asarray(data)
    data = data[-1] if data.ndim == 2 else data
    points = np.linspace(invl.lb.val, invl.ub.val, len(weights))
    return lagrange(points, weights)(data)
