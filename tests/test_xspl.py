import pytest
from xspline.xspl import XSpline


def test_knots_error():
    knots = (1, 1, 1)
    degree = 2
    with pytest.raises(ValueError):
        XSpline(knots, degree)


@pytest.mark.parametrize("degree", [-1, -1.0])
def test_degree_error(degree):
    knots = (0.0, 1.0)
    with pytest.raises(ValueError):
        XSpline(knots, degree)


@pytest.mark.parametrize(
    ("degree", "sdegree", "result"),
    [(2, None, 2), (1, None, 1), (2, 1, 1), (2, 3, 2), (2, -1, -1)],
)
def test_ldegree_rdegree(degree, sdegree, result):
    knots = (0.0, 1.0)
    xspline = XSpline(knots, degree, ldegree=sdegree, rdegree=sdegree)
    assert result == xspline.ldegree and result == xspline.rdegree


def test_len():
    knots = (0.0, 1.0)
    degree = 3
    xspline = XSpline(knots, degree)
    assert len(xspline) == degree + len(knots) - 1
