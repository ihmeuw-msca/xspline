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


@pytest.mark.parametrize("l_linear", [True])
@pytest.mark.parametrize("r_linear", [True])
@pytest.mark.parametrize("idx", [0, -1])
def test_fun(knots, degree, idx, l_linear, r_linear):
    xs = XSpline(knots, degree, l_linear=l_linear, r_linear=r_linear)
    if idx == 0:
        x = np.linspace(xs.knots[0] - 1.0, xs.knots[1], 101)
    else:
        x = np.linspace(xs.knots[-2], xs.knots[-1], 101)
    my_y = xs.fun(x, idx)

    if idx == 0:
        tr_y = (xs.inner_knots[1] - x) / \
               (xs.inner_knots[1] - xs.inner_knots[0])
    else:
        tr_y = (x - xs.inner_knots[-2]) / \
               (xs.inner_knots[-1] - xs.inner_knots[-2])

    assert np.linalg.norm(my_y - tr_y) < 1e-10


@pytest.mark.parametrize("l_linear", [True])
@pytest.mark.parametrize("r_linear", [True])
@pytest.mark.parametrize("order", [1])
@pytest.mark.parametrize("idx", [0, -1])
def test_dfun(knots, degree, order, idx, l_linear, r_linear):
    xs = XSpline(knots, degree, l_linear=l_linear, r_linear=r_linear)
    if idx == 0:
        x = np.linspace(xs.knots[0] - 1.0, xs.knots[1] - 1e-10, 101)
    else:
        x = np.linspace(xs.knots[-2], xs.knots[-1] - 1e-10, 101)
    my_dy = xs.dfun(x, order, idx)

    if idx == 0:
        tr_dy = np.repeat(1.0/(xs.inner_knots[0] - xs.inner_knots[1]), x.size)
    else:
        tr_dy = np.repeat(1.0/(xs.inner_knots[-1] - xs.inner_knots[-2]), x.size)

    assert np.linalg.norm(my_dy - tr_dy) < 1e-10


# @pytest.mark.parametrize("l_linear", [True, False])
# @pytest.mark.parametrize("r_linear", [True, False])
# @pytest.mark.parametrize("order", [1])
# @pytest.mark.parametrize("idx", [0, -1])
# def test_ifun(knots, degree, order, idx, l_linear, r_linear):
#     xs = XSpline(knots, degree, l_linear=l_linear, r_linear=r_linear)
#     if idx == 0:
#         x = np.linspace(xs.knots[0] - 1.0, xs.knots[1], 101)
#     else:
#         x = np.linspace(xs.knots[-2], xs.knots[-1], 101)

#     domain = xs.domain(idx)
#     lb = domain[0]
#     ub = domain[1]
#     v_idx = (x >= lb) & (x <= ub)
#     tr_iy = np.zeros(x.size)

#     if idx == 0:
#         def ifun(a):
#             return 0.5*(xs.inner_knots[1] - a)**2 / \
#                 (xs.inner_knots[0] - xs.inner_knots[1])
#     else:
#         def ifun(a):
#             return 0.5*(a - xs.inner_knots[-2])**2 / \
#                 (xs.inner_knots[-1] - xs.inner_knots[-2])

#     tr_iy[v_idx] = ifun(x[v_idx]) - ifun(lb)


@pytest.mark.parametrize("l_linear", [True, False])
@pytest.mark.parametrize("r_linear", [True, False])
@pytest.mark.parametrize("x", [np.linspace(0.0, 1.0, 100),
                               np.linspace(-1.0, 0.5, 100),
                               np.linspace(0.5, 2.0, 100)])
def test_design_mat(knots, degree, x, l_linear, r_linear):
    xs = XSpline(knots, degree, l_linear=l_linear, r_linear=r_linear)
    mat = xs.design_mat(x)

    assert mat.shape == (x.size, xs.num_spline_bases)


@pytest.mark.parametrize("l_linear", [True, False])
@pytest.mark.parametrize("r_linear", [True, False])
@pytest.mark.parametrize("order", [1])
@pytest.mark.parametrize("x", [np.linspace(0.0, 1.0, 100),
                               np.linspace(-1.0, 0.5, 100),
                               np.linspace(0.5, 2.0, 100)])
def test_design_dmat(knots, degree, order, x, l_linear, r_linear):
    xs = XSpline(knots, degree, l_linear=l_linear, r_linear=r_linear)
    dmat = xs.design_dmat(x, order)

    assert dmat.shape == (x.size, xs.num_spline_bases)


@pytest.mark.parametrize("l_linear", [True, False])
@pytest.mark.parametrize("r_linear", [True, False])
@pytest.mark.parametrize("order", [1])
@pytest.mark.parametrize("x", [np.linspace(0.0, 1.0, 100),
                               np.linspace(-1.0, 0.5, 100),
                               np.linspace(0.5, 2.0, 100)])
def test_design_imat(knots, degree, order, x, l_linear, r_linear):
    xs = XSpline(knots, degree, l_linear=l_linear, r_linear=r_linear)
    imat = xs.design_imat(np.repeat(x[0], x.size), x, order)

    assert imat.shape == (x.size, xs.num_spline_bases)


# @pytest.mark.parametrize("l_linear", [True, False])
# @pytest.mark.parametrize("r_linear", [True, False])
# def test_last_dmat(knots, degree, l_linear, r_linear):
#     xs = XSpline(knots, degree, l_linear=l_linear, r_linear=r_linear)
#     my_last_dmat = xs.last_dmat()

#     x = np.array([0.5*(xs.knots[i] + xs.knots[i+1])
#                   for i in range(xs.knots.size - 1)])
#     tr_last_dmat = xs.design_dmat(x, 1)

#     assert np.linalg.norm(my_last_dmat - tr_last_dmat) < 1e-10
