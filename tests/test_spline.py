"""
Test Spline Module
"""
import pytest
import numpy as np

from xspline.spline import XSpline


@pytest.mark.parametrize("knots", [np.linspace(0.0, 1.0, 5),
                                   np.linspace(0.0, 1.0, 3)])
@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_xspline_design_mat(knots, degree):
    spline = XSpline(knots, degree)
    x = np.linspace(*spline.domain, 5)
    mat = spline.design_mat(x)
    assert np.allclose(mat.sum(axis=1), 1.0)


@pytest.mark.parametrize("knots", [np.linspace(0.0, 1.0, 5),
                                   np.linspace(0.0, 1.0, 3)])
@pytest.mark.parametrize("degree", [3])
@pytest.mark.parametrize("lxorder", [0, 1, 2, 3])
def test_xspline_xorders(knots, degree, lxorder):
    spline = XSpline(knots, degree, lxorder=lxorder)
    x = np.array([spline.domain[0] - 0.1])
    mat = spline.design_mat(x, order=lxorder + 1)
    assert np.allclose(mat, 0.0)
