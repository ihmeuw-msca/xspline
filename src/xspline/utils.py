"""
Utility Modules
"""
from numbers import Number
from collections.abc import Iterable
from typing import Tuple

import numpy as np


def check_fun_input(data: Iterable, order: Number) -> Tuple[np.ndarray, int]:
    data = np.asarray(data)
    data = data[None, :] if data.ndim == 1 else data
    order = int(order)
    assert data.ndim == 2
    assert len(data) in [1, 2]
    assert order >= 0 or len(data) == 2
    return data, order


def taylor_term(data: Iterable, order: Number) -> np.ndarray:
    data, order = check_fun_input(data, order)
    assert order >= 0
    data = data.ravel() if len(data) == 1 else data[1] - data[0]
    return data**order/np.math.factorial(order)
