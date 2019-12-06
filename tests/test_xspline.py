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
    return 3


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
        lb = xs.inner_knots[-2] if r_linear else xs.knots[-2]
    if idx == -1:
        ub = np.inf if r_extra else xs.knots[-1]
    else:
        ub = xs.inner_knots[1] if l_linear else xs.knots[1]

    assert my_domain[0] == lb
    assert my_domain[1] == ub
