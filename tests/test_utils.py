# -*- coding: utf-8 -*-
"""
    test_utils
    ~~~~~~~~~~

    unit tests for xspline.utils
"""
import numpy as np
import pytest
from xspline import utils


@pytest.mark.parametrize(("x", "l_close", "r_close", "result"),
                         [(0.5, False, False, 1.0),
                          (2.0, False, False, 0.0),
                          (1.0, False, False, 0.0),
                          (1.0, False, True, 1.0),
                          (0.0, False, False, 0.0),
                          (0.0, True, False, 1.0),
                          (np.repeat(0.5, 3), False, False, np.ones(3)),
                          (np.repeat(2.0, 3), False, False, np.zeros(3)),
                          (np.repeat(1.0, 3), False, False, np.zeros(3)),
                          (np.repeat(1.0, 3), False, True, np.ones(3)),
                          (np.repeat(0.0, 3), False, False, np.zeros(3)),
                          (np.repeat(0.0, 3), True, False, np.ones(3))])
def test_utils_indicator_f(x, l_close, r_close, result):
    b = np.array([0.0, 1.0])
    my_result = utils.indicator_f(x, b, l_close=l_close, r_close=r_close)
    assert np.linalg.norm(my_result - result) < 1e-10


@pytest.mark.parametrize(("x", "z", "fz", "dfz"),
                         [(1.0, 0.0, -1.0, 2.0),
                          (np.ones(5), 0.0, -1.0, 2.0),
                          (np.ones(5),
                           np.zeros(5), np.repeat(-1.0, 5), np.repeat(2.0, 5)),
                          (1.0,
                           np.zeros(5), np.repeat(-1.0, 5), np.repeat(2.0, 5))])
def test_utils_linear_f(x, z, fz, dfz):
    result = fz + dfz*(x - z)
    my_result = utils.linear_f(x, z, fz, dfz)
    assert np.linalg.norm(my_result - result) < 1e-10


@pytest.mark.parametrize(("x", "b", "result"),
                         [(0.0, np.array([0.0, 1.0]), 0.0),
                          (1.0, np.array([0.0, 1.0]), 1.0),
                          (np.repeat(0.5, 5), np.array([0.0, 1.0]),
                           np.repeat(0.5, 5))])
def test_utils_linear_lf(x, b, result):
    my_result = utils.linear_lf(x, b)
    assert np.linalg.norm(my_result - result) < 1e-10


@pytest.mark.parametrize(("x", "b", "result"),
                         [(0.0, np.array([0.0, 1.0]), 1.0),
                          (1.0, np.array([0.0, 1.0]), 0.0),
                          (np.repeat(0.5, 5), np.array([0.0, 1.0]),
                           np.repeat(0.5, 5))])
def test_utils_linear_rf(x, b, result):
    my_result = utils.linear_rf(x, b)
    assert np.linalg.norm(my_result - result) < 1e-10


@pytest.mark.parametrize("a", [0.0, np.zeros(5)])
@pytest.mark.parametrize("x", [1.0, np.ones(5)])
@pytest.mark.parametrize("order", [0, 1])
def test_utils_constant_if_0(a, x, order):
    my_result = utils.constant_if(a, x, order, 0.0)
    assert np.linalg.norm(my_result) < 1e-10


@pytest.mark.parametrize("a", [0.0, np.zeros(5)])
@pytest.mark.parametrize("x", [1.0, np.ones(5)])
@pytest.mark.parametrize(("order", "result"),
                         [(1, 1.0), (2, 0.5)])
def test_utils_constant_if_1(a, x, order, result):
    my_result = utils.constant_if(a, x, order, 1.0)
    assert np.linalg.norm(my_result - result) < 1e-10


@pytest.mark.parametrize(("z", "fz", "dfz"),
                         [(1.0, 0.0, 2.0)])
@pytest.mark.parametrize("a", [0.0, np.zeros(5)])
@pytest.mark.parametrize("x", [1.0, np.ones(5)])
@pytest.mark.parametrize(("order", "result"),
                         [(0, 0.0), (1, -1.0), (2, -2.0/3.0)])
def test_utils_linear_if(a, x, order, z, fz, dfz, result):
    my_result = utils.linear_if(a, x, order, z, fz, dfz)
    assert np.linalg.norm(my_result - result) < 1e-10


@pytest.mark.parametrize("a", [0.0, np.zeros(5)])
@pytest.mark.parametrize("x", [1.0, np.ones(5)])
@pytest.mark.parametrize("knots", [np.array([0.5])])
@pytest.mark.parametrize(("order", "result"),
                         [(0, 2.0), (1, 1.5), (2, 0.625), (3, 3.0/16.0)])
def test_utils_integrate_across_pieces(a, x, order, knots, result):
    my_result = utils.integrate_across_pieces(
        a, x, order,
        [
            lambda *params: utils.constant_if(*params, 1.0),
            lambda *params: utils.constant_if(*params, 2.0),
        ],
        knots
    )
    assert np.linalg.norm(my_result - result) < 1e-10


@pytest.mark.parametrize("a", [0.0, np.zeros(5)])
@pytest.mark.parametrize("x", [1.0, np.ones(5)])
@pytest.mark.parametrize("knots", [np.array([0.0, 0.5, 1.0])])
@pytest.mark.parametrize(("order", "result"),
                         [(0, 2.0), (1, 1.5), (2, 0.625), (3, 3.0/16.0)])
def test_utils_pieces_if(a, x, order, knots, result):
    my_result = utils.pieces_if(
        a, x, order,
        [
            lambda *params: utils.constant_if(*params, 0.0),
            lambda *params: utils.constant_if(*params, 1.0),
            lambda *params: utils.constant_if(*params, 2.0),
            lambda *params: utils.constant_if(*params, 0.0),
        ],
        knots
    )
    assert np.linalg.norm(my_result - result) < 1e-10


@pytest.mark.parametrize("a", [0.0, np.zeros(5)])
@pytest.mark.parametrize("x", [1.0, np.ones(5)])
@pytest.mark.parametrize("b", [np.array([0.5, 1.0])])
@pytest.mark.parametrize(("order", "result"),
                         [(0, 0.0), (1, 0.5), (2, 0.125)])
def test_utils_indicator_if(a, x, order, b, result):
    my_result = utils.indicator_if(a, x, order, b)
    assert np.linalg.norm(my_result - result) < 1e-10


@pytest.mark.parametrize("size", [5, 10])
def test_utils_seq_diff_mat(size):
    x = np.random.randn(size)
    y = np.array([x[i+1] - x[i] for i in range(size - 1)])

    mat = utils.seq_diff_mat(size)
    my_y = mat.dot(x)

    assert np.linalg.norm(my_y - y) < 1e-10


@pytest.mark.parametrize("order", [0, 2, 4])
@pytest.mark.parametrize("shape", [(5,), (2, 5), (1, 2, 3)])
def test_utils_order_to_index(order, shape):
    x = np.random.randn(*shape)
    z = x.reshape(x.size, 1)
    index = utils.order_to_index(order, shape)
    assert z[order] == x[index]


@pytest.mark.parametrize("option", [True, False, None])
@pytest.mark.parametrize("size", [2, 3])
def test_utils_option_to_list(option, size):
    result = not option
    my_result = utils.option_to_list(option, size)
    assert len(my_result) == size
    assert all([~(my_result[i] ^ result) for i in range(size)])


@pytest.mark.parametrize("args",
                         [(np.arange(2), np.arange(3)),
                          (np.arange(2), np.arange(3), np.arange(4))])
def test_utils_outer_flatten(args):
    result = np.prod(np.ix_(*args))
    result = result.reshape(result.size,)
    my_result = utils.outer_flatten(*args)

    assert np.linalg.norm(my_result - result) < 1e-10
