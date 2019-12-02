# -*- coding: utf-8 -*-
"""
    core
    ~~~~

    core module contains main functions and classes.
"""
import numpy as np
from . import utils


def bspline_domain(knots, degree, idx, l_extra=False, r_extra=False):
    r"""Compute the support for the spline basis, knots degree and the index of
    the basis.

    Args:
        knots (numpy.ndarray):
        1D array that stores the knots of the splines.

        degree (int):
        A non-negative integer that indicates the degree of the polynomial.

        idx (int):
        A non-negative integer that indicates the order in the spline bases
        list.

        l_extra (bool, optional):
        A optional bool variable indicates that if extrapolate at left end.
        Default to be False.

        r_extra (bool, optional):
        A optional bool variable indicates that if extrapolate at right end.
        Default to be False.

    Returns:
        numpy.ndarray:
        1D array with two elements represents that left and right end of the
        support of the spline basis.
    """
    num_knots = knots.size
    num_intervals = num_knots - 1
    num_splines = num_intervals + degree

    if idx == -1:
        idx = num_splines - 1

    lb = knots[max(idx - degree, 0)]
    ub = knots[min(idx + 1, num_intervals)]

    if idx == 0 and l_extra:
        lb = -np.inf
    if idx == num_splines - 1 and r_extra:
        ub = np.inf

    return np.array([lb, ub])


def bspline_fun(x, knots, degree, idx, l_extra=False, r_extra=False):
    r"""Compute the spline basis.

    Args:
        x (float | numpy.ndarray):
        Scalar or numpy array that store the independent variables.

        knots (numpy.ndarray):
        1D array that stores the knots of the splines.

        degree (int):
        A non-negative integer that indicates the degree of the polynomial.

        idx (int):
        A non-negative integer that indicates the order in the spline bases
        list.

        l_extra (bool, optional):
        A optional bool variable indicates that if extrapolate at left end.
        Default to be False.

        r_extra (bool, optional):
        A optional bool variable indicates that if extrapolate at right end.
        Default to be False.

    Returns:
        float | numpy.ndarray:
        Function values of the corresponding spline bases.
    """
    num_knots = knots.size
    num_intervals = num_knots - 1
    num_splines = num_intervals + degree

    if idx == -1:
        idx = num_splines - 1

    b = bspline_domain(knots, degree, idx, l_extra=l_extra, r_extra=r_extra)

    if degree == 0:
        f = utils.indicator_f(x, b, r_close=(idx == num_splines - 1))
        return f

    if idx == 0:
        b_effect = bspline_domain(knots, degree, idx)
        y = utils.indicator_f(x, b)
        z = utils.linear_rf(x, b_effect)
        return y*(z**degree)

    if idx == num_splines - 1:
        b_effect = bspline_domain(knots, degree, idx)
        y = utils.indicator_f(x, b, r_close=True)
        z = utils.linear_lf(x, b_effect)
        return y*(z**degree)

    lf = bspline_fun(x, knots, degree - 1, idx - 1,
                     l_extra=l_extra, r_extra=r_extra)
    lf *= utils.linear_lf(x, bspline_domain(knots, degree - 1, idx - 1))

    rf = bspline_fun(x, knots, degree - 1, idx,
                     l_extra=l_extra, r_extra=r_extra)
    rf *= utils.linear_rf(x, bspline_domain(knots, degree - 1, idx))

    return lf + rf


def bspline_dfun(x, knots, degree, order, idx, l_extra=False, r_extra=False):
    r"""Compute the derivative of the spline basis.

    Args:
        x (float | numpy.ndarray):
        Scalar or numpy array that store the independent variables.

        knots (numpy.ndarray):
        1D array that stores the knots of the splines.

        degree (int):
        A non-negative integer that indicates the degree of the polynomial.

        order (int):
        A non-negative integer that indicates the order of differentiation.

        idx (int):
        A non-negative integer that indicates the order in the spline bases
        list.

        l_extra (bool, optional):
        A optional bool variable indicates that if extrapolate at left end.
        Default to be False.

        r_extra (bool, optional):
        A optional bool variable indicates that if extrapolate at right end.
        Default to be False.

    Returns:
        float | numpy.ndarray:
        Derivative values of the corresponding spline bases.
    """
    num_knots = knots.size
    num_intervals = num_knots - 1
    num_splines = num_intervals + degree

    if idx == -1:
        idx = num_splines - 1

    if order == 0:
        return bspline_fun(x, knots, degree, idx,
                           l_extra=l_extra, r_extra=r_extra)

    if order > degree:
        return np.zeros(x.size)

    if idx == 0:
        rdf = 0.0
    else:
        b = bspline_domain(knots, degree - 1, idx - 1)
        d = b[1] - b[0]
        f = (x - b[0])/d
        rdf = f*bspline_dfun(x, knots, degree - 1, order, idx - 1,
                             l_extra=l_extra, r_extra=r_extra)
        rdf += order*bspline_dfun(x, knots, degree - 1, order - 1, idx - 1,
                                  l_extra=l_extra, r_extra=r_extra)/d

    if idx == num_splines - 1:
        ldf = 0.0
    else:
        b = bspline_domain(knots, degree - 1, idx)
        d = b[0] - b[1]
        f = (x - b[1])/d
        ldf = f*bspline_dfun(x, knots, degree - 1, order, idx,
                             l_extra=l_extra, r_extra=r_extra)
        ldf += order*bspline_dfun(x, knots, degree - 1, order - 1, idx,
                                  l_extra=l_extra, r_extra=r_extra)/d

    return ldf + rdf


def bspline_ifun(a, x, knots, degree, order, idx, l_extra=False, r_extra=False):
    r"""Compute the integral of the spline basis.

    Args:
        x (float | numpy.ndarray):
        Scalar or numpy array that store the independent variables.

        knots (numpy.ndarray):
        1D array that stores the knots of the splines.

        degree (int):
        A non-negative integer that indicates the degree of the polynomial.

        order (int):
        A non-negative integer that indicates the order of integration.

        idx (int):
        A non-negative integer that indicates the order in the spline bases
        list.

        l_extra (bool, optional):
        A optional bool variable indicates that if extrapolate at left end.
        Default to be False.

        r_extra (bool, optional):
        A optional bool variable indicates that if extrapolate at right end.
        Default to be False.

    Returns:
        float | numpy.ndarray:
        Integral values of the corresponding spline bases.
    """
    num_knots = knots.size
    num_intervals = num_knots - 1
    num_splines = num_intervals + degree

    if idx == -1:
        idx = num_splines - 1

    if order == 0:
        return bspline_fun(x, knots, degree, idx,
                           l_extra=l_extra, r_extra=r_extra)

    if degree == 0:
        b = bspline_domain(knots, degree, idx, l_extra=l_extra, r_extra=r_extra)
        return utils.indicator_if(a, x, order, b)

    if idx == 0:
        rif = 0.0
    else:
        b = bspline_domain(knots, degree - 1, idx - 1)
        d = b[1] - b[0]
        f = (x - b[0]) / d
        rif = f*bspline_ifun(a, x, knots, degree - 1, order, idx - 1,
                             l_extra=l_extra, r_extra=r_extra)
        rif -= order*bspline_ifun(a, x, knots, degree - 1, order + 1, idx - 1,
                                  l_extra=l_extra, r_extra=r_extra)/d

    if idx == num_splines - 1:
        lif = 0.0
    else:
        b = bspline_domain(knots, degree - 1, idx)
        d = b[0] - b[1]
        f = (x - b[1]) / d
        lif = f*bspline_ifun(a, x, knots, degree - 1, order, idx,
                             l_extra=l_extra, r_extra=r_extra)
        lif -= order*bspline_ifun(a, x, knots, degree - 1, order + 1, idx,
                                  l_extra=l_extra, r_extra=r_extra)/d

    return lif + rif
