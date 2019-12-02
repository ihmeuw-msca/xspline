# -*- coding: utf-8 -*-
"""
    test
    ~~~~~~~~~

    unit tests for xspline.core
"""
import numpy as np
import pytest
from xspline import core


@pytest.mark.parametrize("knots", [np.linspace(0.0, 1.0, 5)])
@pytest.mark.parametrize("degree", [0, 1, 2])
@pytest.mark.parametrize("idx", [0, 1, -1])
@pytest.mark.parametrize("l_extra", [False, True])
@pytest.mark.parametrize("r_extra", [False, True])
def test_bspline_domain(knots, degree, idx, l_extra, r_extra):
    my_domain = core.bspline_domain(knots, degree, idx,
                                    l_extra=l_extra,
                                    r_extra=r_extra)
    if idx == 0:
        tr_domain = knots[:2].copy()
        if l_extra:
            tr_domain[0] = -np.inf
    elif idx == -1:
        tr_domain = knots[-2:].copy()
        if r_extra:
            tr_domain[1] = np.inf
    elif idx == 1:
        if degree == 0:
            tr_domain = np.array([knots[1], knots[2]])
        else:
            tr_domain = np.array([knots[0], knots[2]])

    assert tr_domain[0] == my_domain[0]
    assert tr_domain[1] == my_domain[1]


@pytest.mark.parametrize("knots", [np.arange(5).astype(float)])
@pytest.mark.parametrize("degree", [0, 1])
def test_bspline_domain_l_extra(knots, degree):
    idx = 0
    result = np.array([-np.inf, knots[1]])
    my_result = core.bspline_domain(knots, degree, idx, l_extra=True)
    assert np.isneginf(my_result[0])
    assert np.abs(my_result[1] - result[1]) < 1e-10


@pytest.mark.parametrize("knots", [np.arange(5).astype(float)])
@pytest.mark.parametrize("degree", [0, 1])
def test_bspline_domain_r_extra(knots, degree):
    idx = knots.size + degree - 2
    result = np.array([knots[-2], np.inf])
    my_result = core.bspline_domain(knots, degree, idx, r_extra=True)
    assert np.isposinf(my_result[1])
    assert np.abs(my_result[0] - result[0]) < 1e-10


@pytest.mark.parametrize("knots", [np.linspace(0.0, 1.0, 5)])
@pytest.mark.parametrize("degree", [1])
@pytest.mark.parametrize("idx", [0, 1])
@pytest.mark.parametrize("x", [np.linspace(0.0, 1.0, 101)])
def test_bspline_fun(knots, degree, idx, x):
    my_y = core.bspline_fun(x, knots, degree, idx)
    if idx == 0:
        tr_y = np.maximum((knots[1] - x)/knots[1], 0.0)
    else:
        tr_y = np.zeros(x.size)
        idx1 = (x >= knots[0]) & (x < knots[1])
        idx2 = (x >= knots[1]) & (x < knots[2])
        tr_y[idx1] = x[idx1]/knots[1]
        tr_y[idx2] = (knots[2] - x[idx2])/(knots[2] - knots[1])

    assert np.linalg.norm(tr_y - my_y) < 1e-10


@pytest.mark.parametrize("knots", [np.linspace(0.0, 1.0, 5)])
@pytest.mark.parametrize("x", [np.linspace(-1.0, -0.1, 10)])
@pytest.mark.parametrize("l_extra", [True, False])
def test_bspline_l_extra(x, knots, l_extra):
    degree = 0
    idx = 0
    my_y = core.bspline_fun(x, knots, degree, idx, l_extra=l_extra)
    if l_extra:
        tr_y = 1.0
    else:
        tr_y = 0.0
    assert np.linalg.norm(my_y - tr_y) < 1e-10


@pytest.mark.parametrize("knots", [np.linspace(0.0, 1.0, 5)])
@pytest.mark.parametrize("x", [np.linspace(1.1, 2.0, 10)])
@pytest.mark.parametrize("r_extra", [True, False])
def test_bspline_r_extra(x, knots, r_extra):
    degree = 0
    idx = -1
    my_y = core.bspline_fun(x, knots, degree, idx, r_extra=r_extra)
    if r_extra:
        tr_y = 1.0
    else:
        tr_y = 0.0
    assert np.linalg.norm(my_y - tr_y) < 1e-10
