"""
Test Functions Module
"""
import pytest
import numpy as np
from xspline.interval import Interval
from xspline.funs import ConstFunction, IndicatorFunction


@pytest.mark.parametrize("const", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("data", [np.ones(3), np.zeros(4), np.random.randn(5)])
def test_con_fun(const, data):
    const_fun = ConstFunction(const=const)
    assert np.allclose(const_fun(data), const)


@pytest.mark.parametrize("const", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("data", [np.ones(3), np.zeros(4), np.random.randn(5)])
def test_con_dfun(const, data):
    const_fun = ConstFunction(const=const)
    assert np.allclose(const_fun(data, order=1), 0.0)


@pytest.mark.parametrize("const", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("data", [np.vstack([np.zeros(3), np.random.randn(3)])])
def test_con_ifun(const, data):
    const_fun = ConstFunction(const=const)
    assert np.allclose(const_fun(data, order=-1), const*data[-1])


@pytest.mark.parametrize(("invl", "val"),
                         [(Interval(0.0, 1.0), [0.0, 0.0, 1.0, 1.0, 0.0]),
                          (Interval(-np.inf, 0.0), [1.0, 1.0, 1.0, 0.0, 0.0]),
                          (Interval(1.0, np.inf), [0.0, 0.0, 0.0, 1.0, 1.0]),
                          (Interval(-np.inf, np.inf), [1.0, 1.0, 1.0, 1.0, 1.0])])
@pytest.mark.parametrize("data", [[-2.0, -1.0, 0.0, 1.0, 2.0]])
def test_ind_fun(invl, val, data):
    indic_fun = IndicatorFunction(domain=invl)
    assert np.allclose(indic_fun(data), val)


@pytest.mark.parametrize("invl",
                         [Interval(0.0, 1.0),
                          Interval(-np.inf, 0.0),
                          Interval(1.0, np.inf),
                          Interval(-np.inf, np.inf)])
@pytest.mark.parametrize("data", [[-2.0, -1.0, 0.0, 1.0, 2.0]])
def test_ind_dfun(invl, data):
    indic_fun = IndicatorFunction(domain=invl)
    assert np.allclose(indic_fun(data, order=1), 0.0)


@pytest.mark.parametrize(("invl", "val"),
                         [(Interval(0.0, 1.0), [1.0, 1.0, 1.0, 1.0, 1.0]),
                          (Interval(-np.inf, 0.0), [0.0]*5),
                          (Interval(1.0, np.inf), [0.0, 1.0, 2.0, 3.0, 4.0]),
                          (Interval(-np.inf, np.inf), [1.0, 2.0, 3.0, 4.0, 5.0])])
@pytest.mark.parametrize("data", [[[0.0]*5, [1.0, 2.0, 3.0, 4.0, 5.0]]])
def test_ind_ifun(invl, val, data):
    indic_fun = IndicatorFunction(domain=invl)
    assert np.allclose(indic_fun(data, order=-1), val)
