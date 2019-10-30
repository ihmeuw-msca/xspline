# -*- coding: utf-8 -*-
"""
    test_core
    ~~~~~~~~~

    unit tests for xspline.core
"""
import numpy as np
import pytest
from xspline import core


@pytest.mark.parametrize("knots", [np.arange(5).astype(float)])
@pytest.mark.parametrize(("degree", "idx", "result"),
                         [(0, 0, np.array([0.0, 1.0])),
                          (0, 1, np.array([1.0, 2.0])),
                          (1, 0, np.array([0.0, 1.0])),
                          (1, 1, np.array([0.0, 2.0]))])
def test_core_bspline_b(knots, degree, idx, result):
    my_result = core.bspline_b(knots, degree, idx)
    assert np.linalg.norm(my_result - result) < 1e-10


@pytest.mark.parametrize("knots", [np.arange(5).astype(float)])
@pytest.mark.parametrize("degree", [0, 1])
def test_core_bspline_b_l_extra(knots, degree):
    idx = 0
    result = np.array([-np.inf, knots[1]])
    my_result = core.bspline_b(knots, degree, idx, l_extra=True)
    assert np.isneginf(my_result[0])
    assert np.abs(my_result[1] - result[1]) < 1e-10


@pytest.mark.parametrize("knots", [np.arange(5).astype(float)])
@pytest.mark.parametrize("degree", [0, 1])
def test_core_bspline_b_r_extra(knots, degree):
    idx = knots.size + degree - 2
    result = np.array([knots[-2], np.inf])
    my_result = core.bspline_b(knots, degree, idx, r_extra=True)
    assert np.isposinf(my_result[1])
    assert np.abs(my_result[0] - result[0]) < 1e-10
