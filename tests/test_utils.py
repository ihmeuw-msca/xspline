# -*- coding: utf-8 -*-
"""
    test_utils
    ~~~~~~~~~~

    unit tests for xspline.utils
"""
import numpy as np
import pytest
from xspline import utils


@pytest.mark.parametrize(("x", "l_close", "r_close", "result"),
                         [(0.5, False, False, 1.0),
                          (2.0, False, False, 0.0),
                          (1.0, False, False, 0.0),
                          (1.0, False, True, 1.0),
                          (0.0, False, False, 0.0),
                          (0.0, True, False, 1.0),
                          (np.repeat(0.5, 3), False, False, np.ones(3)),
                          (np.repeat(2.0, 3), False, False, np.zeros(3)),
                          (np.repeat(1.0, 3), False, False, np.zeros(3)),
                          (np.repeat(1.0, 3), False, True, np.ones(3)),
                          (np.repeat(0.0, 3), False, False, np.zeros(3)),
                          (np.repeat(0.0, 3), True, False, np.ones(3))])
def test_utils_indicator_f(x, l_close, r_close, result):
    b = np.array([0.0, 1.0])
    my_result = utils.indicator_f(x, b, l_close=l_close, r_close=r_close)
    assert np.linalg.norm(my_result - result) < 1e-10


@pytest.mark.parametrize(("x", "z", "fz", "dfz"),
                         [(1.0, 0.0, -1.0, 2.0),
                          (np.ones(5), 0.0, -1.0, 2.0),
                          (np.ones(5),
                           np.zeros(5), np.repeat(-1.0, 5), np.repeat(2.0, 5)),
                          (1.0,
                           np.zeros(5), np.repeat(-1.0, 5), np.repeat(2.0, 5))])
def test_utils_linear_f(x, z, fz, dfz):
    result = fz + dfz*(x - z)
    my_result = utils.linear_f(x, z, fz, dfz)
    assert np.linalg.norm(my_result - result) < 1e-10


@pytest.mark.parametrize(("x", "b", "result"),
                         [(0.0, np.array([0.0, 1.0]), 0.0),
                          (1.0, np.array([0.0, 1.0]), 1.0),
                          (np.repeat(0.5, 5), np.array([0.0, 1.0]),
                           np.repeat(0.5, 5))])
def test_utils_linear_lf(x, b, result):
    my_result = utils.linear_lf(x, b)
    assert np.linalg.norm(my_result - result) < 1e-10


@pytest.mark.parametrize(("x", "b", "result"),
                         [(0.0, np.array([0.0, 1.0]), 1.0),
                          (1.0, np.array([0.0, 1.0]), 0.0),
                          (np.repeat(0.5, 5), np.array([0.0, 1.0]),
                           np.repeat(0.5, 5))])
def test_utils_linear_rf(x, b, result):
    my_result = utils.linear_rf(x, b)
    assert np.linalg.norm(my_result - result) < 1e-10


@pytest.mark.parametrize("a", [0.0, np.zeros(5)])
@pytest.mark.parametrize("x", [1.0, np.ones(5)])
@pytest.mark.parametrize("order", [0, 1])
def test_utils_constant_if_0(a, x, order):
    my_result = utils.constant_if(a, x, order, 0.0)
    assert np.linalg.norm(my_result) < 1e-10


@pytest.mark.parametrize("a", [0.0, np.zeros(5)])
@pytest.mark.parametrize("x", [1.0, np.ones(5)])
@pytest.mark.parametrize(("order", "result"),
                         [(1, 1.0), (2, 0.5)])
def test_utils_constant_if_1(a, x, order, result):
    my_result = utils.constant_if(a, x, order, 1.0)
    assert np.linalg.norm(my_result - result) < 1e-10
