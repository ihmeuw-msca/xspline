import numpy as np
import pytest
from xspline.indi import Indi
from xspline.xfunction import BasisXFunction


@pytest.fixture
def xfun():
    fun0 = Indi(((0.0, True), (1.0, True)))
    fun1 = Indi(((1.5, True), (2.5, True)))

    return BasisXFunction([fun0, fun1])


def test_len(xfun):
    assert len(xfun) == 2


def test_coef_error(xfun):
    coef = [1, 2, 3]
    with pytest.raises(ValueError):
        xfun.coef = coef


@pytest.mark.parametrize("order", [-1, 0, 1])
def test_get_design_mat(xfun, order):
    x = np.linspace(-0.5, 3.0, 101)
    design_mat = xfun.get_design_mat(x, order=order)
    assert design_mat.shape == (x.size, len(xfun))


@pytest.mark.parametrize("order", [-1, 0, 1])
def test_fun(xfun, order):
    xfun.coef = [1, 1]
    x = np.linspace(-0.5, 3.0, 101)
    result = xfun(x, order=order)
    assert result.shape == x.shape


def test_fun_error(xfun):
    x = np.linspace(-0.5, 3.0, 101)
    with pytest.raises(ValueError):
        xfun(x)
