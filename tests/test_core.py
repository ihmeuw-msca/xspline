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
