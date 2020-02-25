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


@pytest.mark.parametrize('l_extra_list', [None,
                                          [False, False],
                                          [True, False],
                                          [False, True],
                                          [True, True]])
@pytest.mark.parametrize('r_extra_list', [None,
                                          [False, False],
                                          [True, False],
                                          [False, True],
                                          [True, True]])
@pytest.mark.parametrize('is_grid', [True, False])
def test_design_mat(spline, is_grid, l_extra_list, r_extra_list):
    x = np.linspace(0.0, 1.0, 11)
    y = np.linspace(0.0, 1.0, 11)
    mat = spline.design_mat([x, y],
                            is_grid=is_grid,
                            l_extra_list=l_extra_list,
                            r_extra_list=r_extra_list)
    if is_grid:
        assert mat.shape == (x.size*y.size, spline.num_spline_bases)
    else:
        assert mat.shape == (x.size, spline.num_spline_bases)


@pytest.mark.parametrize('l_extra_list', [None,
                                          [False, False],
                                          [True, False],
                                          [False, True],
                                          [True, True]])
@pytest.mark.parametrize('r_extra_list', [None,
                                          [False, False],
                                          [True, False],
                                          [False, True],
                                          [True, True]])
@pytest.mark.parametrize('is_grid', [True, False])
def test_design_dmat(spline, is_grid, l_extra_list, r_extra_list):
    x = np.linspace(0.0, 1.0, 11)
    y = np.linspace(0.0, 1.0, 11)
    dmat = spline.design_dmat([x, y], [1, 1],
                              is_grid=is_grid,
                              l_extra_list=l_extra_list,
                              r_extra_list=r_extra_list)
    if is_grid:
        assert dmat.shape == (x.size*y.size, spline.num_spline_bases)
    else:
        assert dmat.shape == (x.size, spline.num_spline_bases)


@pytest.mark.parametrize('l_extra_list', [None,
                                          [False, False],
                                          [True, False],
                                          [False, True],
                                          [True, True]])
@pytest.mark.parametrize('r_extra_list', [None,
                                          [False, False],
                                          [True, False],
                                          [False, True],
                                          [True, True]])
@pytest.mark.parametrize('is_grid', [True, False])
def test_design_imat(spline, is_grid, l_extra_list, r_extra_list):
    x = np.linspace(0.0, 1.0, 11)
    y = np.linspace(0.0, 1.0, 11)
    imat = spline.design_imat([x, y], [x, y], [1, 1],
                              is_grid=is_grid,
                              l_extra_list=l_extra_list,
                              r_extra_list=r_extra_list)
    assert np.allclose(imat, 0.0)
    if is_grid:
        assert imat.shape == (x.size*y.size, spline.num_spline_bases)
    else:
        assert imat.shape == (x.size, spline.num_spline_bases)