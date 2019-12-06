# -*- coding: utf-8 -*-
"""
    test_xspline
    ~~~~~~~~~~~~

    Test XSpline class.
"""
import numpy as np
import pytest
from xspline.core import XSpline


@pytest.fixture
def knots():
    return np.linspace(0.0, 1.0, 5)


@pytest.fixture
def degree():
    return 1


@pytest.mark.parametrize("l_linear", [True, False])
@pytest.mark.parametrize("r_linear", [True, False])
@pytest.mark.parametrize("l_extra", [True, False])
@pytest.mark.parametrize("r_extra", [True, False])
@pytest.mark.parametrize("idx", [0, -1])
def test_domain(knots, degree, idx, l_linear, r_linear, l_extra, r_extra):
    xs = XSpline(knots, degree, l_linear=l_linear, r_linear=r_linear)
    my_domain = xs.domain(idx, l_extra=l_extra, r_extra=r_extra)
    if idx == 0:
        lb = -np.inf if l_extra else xs.knots[0]
    else:
        lb = xs.inner_knots[-2]
    if idx == -1:
        ub = np.inf if r_extra else xs.knots[-1]
    else:
        ub = xs.inner_knots[1]

    assert my_domain[0] == lb
    assert my_domain[1] == ub


@pytest.mark.parametrize("l_linear", [True, False])
@pytest.mark.parametrize("r_linear", [True, False])
@pytest.mark.parametrize("l_extra", [True, False])
@pytest.mark.parametrize("r_extra", [True, False])
@pytest.mark.parametrize("idx", [0, -1])
def test_fun(knots, degree, idx, l_linear, r_linear, l_extra, r_extra):
    xs = XSpline(knots, degree, l_linear=l_linear, r_linear=r_linear)
    if idx == 0:
        x = np.linspace(xs.knots[0] - 1.0, xs.knots[1], 101)
    else:
        x = np.linspace(xs.knots[-2], xs.knots[-1], 101)
    my_y = xs.fun(x, idx, l_extra=l_extra, r_extra=r_extra)

    if idx == 0:
        tr_y = (xs.inner_knots[1] - x) / \
               (xs.inner_knots[1] - xs.inner_knots[0])
        if not l_extra:
            tr_y[x < xs.knots[0]] = 0.0
    else:
        tr_y = (x - xs.inner_knots[-2]) / \
               (xs.inner_knots[-1] - xs.inner_knots[-2])
        if not r_extra:
            tr_y[x > xs.knots[-1]] = 0.0

    assert np.linalg.norm(my_y - tr_y) < 1e-10
