import numpy as np
import pytest
from numpy.typing import NDArray
from xspline.eval import poly_eval

# define the parameters
params = [(12, -6, 2)]
x = [np.linspace(-1, 2, 101)]
X = [np.vstack([np.ones(x[0].size, dtype=x[0].dtype), x[0]])]
p = [-2, -1, 0, 1, 2]


# define the true functions
def true_eval(x: NDArray, p: int) -> NDArray:
    if p == 0:
        return 12*x**2 - 6*x + 2
    if p == 1:
        return 24*x - 6
    if p == 2:
        return np.repeat(24, x.size)
    if p == -1:
        return 4*x**3 - 3*x**2 + 2*x - 3
    if p == -2:
        return x**4 - x**3 + x**2 - 3*x + 2
    raise ValueError("not support value for 'p'")


@pytest.mark.parametrize("params", params)
@pytest.mark.parametrize("x", x)
@pytest.mark.parametrize("X", X)
@pytest.mark.parametrize("p", p)
def test_poly_eval(params, x, X, p):
    assert np.allclose(
        poly_eval(params, X, p),
        true_eval(x, p)
    )
