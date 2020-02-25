# -*- coding: utf-8 -*-
"""
    test_xspline
    ~~~~~~~~~~~~

    Test XSpline class.
"""
import numpy as np
import pytest
from xspline import NDXSpline

@pytest.fixture
def spline():
    knots = [
        np.linspace(0.0, 1.0, 3),
        np.linspace(0.0, 1.0, 2)
    ]
    degree = [2, 3]
    return NDXSpline(2, knots, degree)


def test_ndim(spline):
    assert spline.ndim == 2


def test_num_knots(spline):
    assert all([spline.num_knots_list[i] == spline.knots_list[i].size
                for i in range(spline.ndim)])
    assert spline.num_knots == np.prod(spline.num_knots_list)


def test_num_intervals(spline):
    assert all([spline.num_intervals_list[i] == spline.knots_list[i].size - 1
                for i in range(spline.ndim)])
    assert spline.num_intervals == np.prod(spline.num_intervals_list)


def test_num_spline_bases(spline):
    assert all([spline.num_spline_bases_list[i] ==
                spline.spline_list[i].num_spline_bases
                for i in range(spline.ndim)])
    assert spline.num_spline_bases == np.prod(spline.num_spline_bases_list)
