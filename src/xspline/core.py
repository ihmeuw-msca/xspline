# -*- coding: utf-8 -*-
"""
    core
    ~~~~

    core module contains main functions and classes.
"""
import numpy as np
from xspline import utils


def bspline_b(knots, degree, idx, l_extra=False, r_extra=False):
    num_knots = knots.size
    num_intervals = num_knots - 1
    num_splines = num_intervals + degree

    lb = knots[max(idx - degree, 0)]
    ub = knots[min(idx + 1, num_intervals)]

    if idx == 0 and l_extra:
        lb = -np.inf
    if idx == num_splines - 1 and r_extra:
        ub = np.inf

    return [lb, ub]


def bspline_f(x, knots, degree, idx, l_extra=False, r_extra=False):
    num_knots = knots.size
    num_intervals = num_knots - 1
    num_splines = num_intervals + degree

    b = bspline_b(idx, degree, l_extra=l_extra, r_extra=r_extra)

    if degree == 0:
        f = utils.indicator_f(x, b, r_close=(idx == num_splines - 1))
        return f

    if idx == 0:
        b_effect = bspline_b(idx, degree)
        y = utils.indicator_f(x, b)
        z = utils.linear_rf(x, b_effect)
        return y*(z**degree)

    if idx == num_splines - 1:
        b_effect = bspline_b(idx, degree)
        y = utils.indicator_f(x, b, r_close=True)
        z = utils.linear_lf(x, b_effect)
        return y*(z**degree)

    lf = bspline_f(x, knots, degree - 1, idx - 1,
                   l_extra=l_extra, r_extra=r_extra)
    lf *= utils.linear_lf(x, bspline_b(knots, degree - 1, idx - 1))

    rf = bspline_f(x, knots, degree - 1, idx,
                   l_extra=l_extra, r_extra=r_extra)
    rf *= utils.linear_rf(x, bspline_b(knots, degree - 1, idx))

    return lf + rf


def bspline_df(x, knots, degree, order, idx, l_extra=False, r_extra=False):
    num_knots = knots.size
    num_intervals = num_knots - 1
    num_splines = num_intervals + degree

    if order == 0:
        return bspline_f(x, knots, degree, idx,
                         l_extra=l_extra, r_extra=r_extra)

    if order > degree:
        return np.zeros(x.size)

    if idx == 0:
        rdf = 0.0
    else:
        b = bspline_b(knots, degree - 1, idx - 1)
        d = b[1] - b[0]
        f = (x - b[0])/d
        rdf = f*bspline_df(x, knots, degree - 1, order, idx - 1,
                           l_extra=l_extra, r_extra=r_extra)
        rdf += order*bspline_df(x, knots, degree - 1, order - 1, idx - 1,
                                l_extra=l_extra, r_extra=r_extra)/d

    if idx == num_splines - 1:
        ldf = 0.0
    else:
        b = bspline_b(knots, degree - 1, idx)
        d = b[0] - b[1]
        f = (x - b[1])/d
        ldf = f*bspline_df(x, knots, degree - 1, order, idx,
                           l_extra=l_extra, r_extra=r_extra)
        ldf += order*bspline_df(x, knots, degree - 1, order - 1, idx,
                                l_extra=l_extra, r_extra=r_extra)/d

    return ldf + rdf


def bspline_if(a, x, knots, degree, order, idx, l_extra=False, r_extra=False):
    num_knots = knots.size
    num_intervals = num_knots - 1
    num_splines = num_intervals + degree

    if order == 0:
        return bspline_f(x, knots, degree, idx,
                         l_extra=l_extra, r_extra=r_extra)

    if degree == 0:
        b = bspline_b(knots, degree, idx, l_extra=l_extra, r_extra=r_extra)
        return utils.indicator_if(a, x, order, b)

    if idx == 0:
        rif = 0.0
    else:
        b = bspline_b(knots, degree - 1, idx - 1)
        d = b[1] - b[0]
        f = (x - b[0]) / d
        rif = f*bspline_if(a, x, knots, degree - 1, order, idx - 1,
                           l_extra=l_extra, r_extra=r_extra)
        rif -= order*bspline_if(a, x, knots, degree - 1, order + 1, idx - 1,
                                l_extra=l_extra, r_extra=r_extra)/d

    if idx == num_splines - 1:
        lif = 0.0
    else:
        b = bspline_b(knots, degree - 1, idx)
        d = b[0] - b[1]
        f = (x - b[1]) / d
        lif = f*bspline_if(a, x, knots, degree - 1, order, idx,
                           l_extra=l_extra, r_extra=r_extra)
        lif -= order*bspline_if(a, x, knots, degree - 1, order + 1, idx,
                                l_extra=l_extra, r_extra=r_extra)/d

    return lif + rif
