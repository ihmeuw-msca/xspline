"""
Function Utility Modules
"""
from collections.abc import Iterable
from numbers import Number
from typing import Tuple, Type, Union

import numpy as np
from numpy.polynomial.polynomial import polyvander
from scipy.special import comb

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


def shift_poly(coefs: np.ndarray, offset: Union[float, np.ndarray]) -> np.ndarray:
    n = len(coefs)
    coefs_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n - i):
            coefs_mat[i, j] = coefs[j + i]*comb(j + i, j)
    offset_mat = polyvander(offset, n - 1)
    return offset_mat.dot(coefs_mat)
