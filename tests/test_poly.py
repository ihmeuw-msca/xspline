import numpy as np
import pytest
from numpy.typing import NDArray
from xspline.poly import Poly

# define the parameters
params = [(12, -6, 2)]
x = [np.linspace(1.0, 2, 101)]
order = [-2, -1, 0, 1, 2]


def truth(x: NDArray, order: int) -> NDArray:
    if order == 0:
        return 12 * x**2 - 6 * x + 2
    if order == 1:
        return 24 * x - 6
    if order == 2:
        return np.repeat(24, x.size)
    if order == -1:
        return 4 * x**3 - 3 * x**2 + 2 * x - 3
    if order == -2:
        return x**4 - x**3 + x**2 - 3 * x + 2
    raise ValueError("not support value for 'order'")


@pytest.mark.parametrize("params", params)
@pytest.mark.parametrize("x", x)
@pytest.mark.parametrize("order", order)
def test_poly(params, x, order):
    poly = Poly(params)
    z = x
    if order < 0:
        z = np.vstack([np.ones(x.size), x])
    assert np.allclose(poly(z, order), truth(x, order))
