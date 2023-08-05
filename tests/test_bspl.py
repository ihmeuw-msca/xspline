import numpy as np
import pytest
from xspline.bspl import Bspl, clear_bspl_cache

knots = (0.0, 0.5, 1.0)
degree = 2
x = np.linspace(knots[0], knots[1], 101)
poly_params = [
    [(4.0, -4.0, 1.0), (0.0, 0.0, 0.0)],
    [(-6.0, 4.0, 0.0), (2.0, -4.0, 2.0)],
    [(2.0, 0.0, 0.0), (-6.0, 8.0, -2.0)],
    [(0.0, 0.0, 0.0), (4.0, -4.0, 1.0)],
]
indices = list(range(-degree, len(knots) - 1))


@pytest.mark.parametrize("index", indices)
def test_bspl_val(index):
    bspl = Bspl((knots, degree, index))
    sub_poly_params = poly_params[index + degree]

    my_val = bspl(x)
    tr_val = np.where(
        x < knots[1],
        np.polyval(sub_poly_params[0], x),
        np.polyval(sub_poly_params[1], x),
    )
    assert np.allclose(my_val, tr_val)

    clear_bspl_cache()


@pytest.mark.parametrize("index", indices)
def test_bspl_der(index):
    bspl = Bspl((knots, degree, index))
    sub_poly_params = list(map(lambda c: np.polyder(c, 1), poly_params[index + degree]))

    my_val = bspl(x, order=1)
    tr_val = np.where(
        x < knots[1],
        np.polyval(sub_poly_params[0], x),
        np.polyval(sub_poly_params[1], x),
    )
    assert np.allclose(my_val, tr_val)

    clear_bspl_cache()


@pytest.mark.parametrize("index", indices)
def test_bspl_int(index):
    bspl = Bspl((knots, degree, index))
    sub_poly_params = list(map(lambda c: np.polyint(c, 1), poly_params[index + degree]))
    offsets = [
        -np.polyval(sub_poly_params[0], 0.25),
        -np.polyval(sub_poly_params[1], knots[1])
        + (
            np.polyval(sub_poly_params[0], knots[1])
            - np.polyval(sub_poly_params[0], 0.25)
        ),
    ]
    a = 0.25
    x = np.vstack([np.repeat(a, 76), np.linspace(a, knots[2], 76)])
    my_val = bspl(x, order=-1)
    tr_val = np.where(
        x[1] < knots[1],
        np.polyval(sub_poly_params[0], x[1]) + offsets[0],
        np.polyval(sub_poly_params[1], x[1]) + offsets[1],
    )
    assert np.allclose(my_val, tr_val)

    clear_bspl_cache()
