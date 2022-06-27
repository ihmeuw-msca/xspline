import numpy as np
import pytest
from xspline import core


@pytest.mark.parametrize("knots", [np.linspace(0.0, 1.0, 5)])
@pytest.mark.parametrize("degree", [0, 1, 2])
@pytest.mark.parametrize("idx", [0, 1, -1])
def test_bspline_domain(knots, degree, idx):
    my_domain = core.bspline_domain(knots, degree, idx)
    if idx == 0:
        tr_domain = knots[:2].copy()
        tr_domain[0] = -np.inf
    elif idx == -1:
        tr_domain = knots[-2:].copy()
        tr_domain[1] = np.inf
    elif idx == 1:
        if degree == 0:
            tr_domain = np.array([knots[1], knots[2]])
        else:
            tr_domain = np.array([knots[0], knots[2]])

    assert tr_domain[0] == my_domain[0]
    assert tr_domain[1] == my_domain[1]


# @pytest.mark.parametrize("knots", [np.linspace(0.0, 1.0, 5)])
# @pytest.mark.parametrize("degree", [1])
# @pytest.mark.parametrize("idx", [0, 1])
# @pytest.mark.parametrize("x", [np.linspace(0.0, 1.0, 101)])
# def test_bspline_fun(x, knots, degree, idx):
#     my_y = core.bspline_fun(x, knots, degree, idx)
#     if idx == 0:
#         tr_y = np.maximum((knots[1] - x)/knots[1], 0.0)
#     else:
#         tr_y = np.zeros(x.size)
#         idx1 = (x >= knots[0]) & (x < knots[1])
#         idx2 = (x >= knots[1]) & (x < knots[2])
#         tr_y[idx1] = x[idx1]/knots[1]
#         tr_y[idx2] = (knots[2] - x[idx2])/(knots[2] - knots[1])
#     assert np.linalg.norm(tr_y - my_y) < 1e-10


# @pytest.mark.parametrize("knots", [np.linspace(0.0, 1.0, 5)])
# @pytest.mark.parametrize("degree", [1])
# @pytest.mark.parametrize("order", [1])
# @pytest.mark.parametrize("idx", [0, 1])
# @pytest.mark.parametrize("x", [np.linspace(0.0, 1.0, 101)])
# def test_bspline_dfun(x, knots, degree, order, idx):
#     my_dy = core.bspline_dfun(x, knots, degree, order, idx)
#     tr_dy = np.zeros(x.size)
#     idx1 = (x >= knots[0]) & (x < knots[1])
#     idx2 = (x >= knots[1]) & (x < knots[2])
#     if idx == 0:
#         tr_dy[idx1] = -1.0/knots[1]
#     else:
#         tr_dy[idx1] = 1.0/knots[1]
#         tr_dy[idx2] = -1.0/(knots[2] - knots[1])

#     assert np.linalg.norm(tr_dy - my_dy) < 1e-10


# @pytest.mark.parametrize("knots", [np.linspace(0.0, 1.0, 5)])
# @pytest.mark.parametrize("degree", [1])
# @pytest.mark.parametrize("order", [1])
# @pytest.mark.parametrize("idx", [0])
# @pytest.mark.parametrize("x", [np.linspace(0.0, 1.0, 101)])
# def test_bspline_ifun(x, knots, degree, order, idx):
#     my_iy = core.bspline_ifun(knots[0], x, knots, degree, order, idx)
#     tr_iy = np.zeros(x.size)
#     idx1 = (x >= knots[0]) & (x <= knots[1])

#     tr_iy[idx1] = x[idx1] - 0.5/knots[1]*x[idx1]**2
#     tr_iy[~idx1] = tr_iy[idx1][-1]

#     assert np.linalg.norm(tr_iy - my_iy) < 1e-10
