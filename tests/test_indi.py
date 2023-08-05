import numpy as np
import pytest
from numpy.typing import NDArray
from xspline.indi import Indi

# define parameters
params = [((1.0, True), (2.0, False))]
x = [np.linspace(1, 2, 101)]
order = [-2, -1, 0, 1, 2]


def truth(x: NDArray, order: int) -> NDArray:
    result = np.zeros(x.size)
    index0 = (x >= 1.0) & (x < 2)
    index1 = x >= 2.0
    if order == 0:
        result[index0] = 1.0
        return result
    if order == 1:
        return result
    if order == 2:
        return result
    if order == -1:
        result[index0] = x[index0] - 1.0
        result[index1] = 1.0
        return result
    if order == -2:
        result[index0] = 0.5 * x[index0] ** 2 - x[index0] + 0.5
        result[index1] = x[index1] - 1.5
        return result
    raise ValueError("not support value for 'order'")


@pytest.mark.parametrize("params", params)
@pytest.mark.parametrize("x", x)
@pytest.mark.parametrize("order", order)
def test_indi(params, x, order):
    indi = Indi(params)
    z = x
    if order < 0:
        z = np.vstack([np.ones(x.size), x])
    assert np.allclose(indi(z, order), truth(x, order))
