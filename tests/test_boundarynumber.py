"""
Test Boundary Number Class
"""
import pytest
from xspline.interval import BoundaryNumber


def test_default():
    a = BoundaryNumber(1)
    assert a.cld


@pytest.mark.parametrize("a_val", [1, 2])
@pytest.mark.parametrize("b_val", [1, 2])
@pytest.mark.parametrize("a_cld", [0, 1])
@pytest.mark.parametrize("b_cld", [0, 1])
def test_eq(a_val, b_val, a_cld, b_cld):
    a = BoundaryNumber(a_val, a_cld)
    b = BoundaryNumber(b_val, b_cld)
    anum = 10*a.val + a.cld
    bnum = 10*b.val + b.cld
    assert (a == b) == (anum == bnum)


@pytest.mark.parametrize("a_val", [1, 2])
@pytest.mark.parametrize("b_val", [1, 2])
@pytest.mark.parametrize("a_cld", [0, 1])
@pytest.mark.parametrize("b_cld", [0, 1])
def test_lt(a_val, b_val, a_cld, b_cld):
    a = BoundaryNumber(a_val, a_cld)
    b = BoundaryNumber(b_val, b_cld)
    anum = 10*a.val + a.cld
    bnum = 10*b.val + b.cld
    assert (a < b) == (anum < bnum)


@pytest.mark.parametrize("a_val", [1, 2])
@pytest.mark.parametrize("b_val", [1, 2])
@pytest.mark.parametrize("a_cld", [0, 1])
@pytest.mark.parametrize("b_cld", [0, 1])
def test_le(a_val, b_val, a_cld, b_cld):
    a = BoundaryNumber(a_val, a_cld)
    b = BoundaryNumber(b_val, b_cld)
    anum = 10*a.val + a.cld
    bnum = 10*b.val + b.cld
    assert (a <= b) == (anum <= bnum)


@pytest.mark.parametrize("a_val", [1, 2])
@pytest.mark.parametrize("b_val", [1, 2])
@pytest.mark.parametrize("a_cld", [0, 1])
@pytest.mark.parametrize("b_cld", [0, 1])
def test_gt(a_val, b_val, a_cld, b_cld):
    a = BoundaryNumber(a_val, a_cld)
    b = BoundaryNumber(b_val, b_cld)
    anum = 10*a.val + a.cld
    bnum = 10*b.val + b.cld
    assert (a > b) == (anum > bnum)


@pytest.mark.parametrize("a_val", [1, 2])
@pytest.mark.parametrize("b_val", [1, 2])
@pytest.mark.parametrize("a_cld", [0, 1])
@pytest.mark.parametrize("b_cld", [0, 1])
def test_ge(a_val, b_val, a_cld, b_cld):
    a = BoundaryNumber(a_val, a_cld)
    b = BoundaryNumber(b_val, b_cld)
    anum = 10*a.val + a.cld
    bnum = 10*b.val + b.cld
    assert (a >= b) == (anum >= bnum)


@pytest.mark.parametrize(("a", "b"),
                         [(BoundaryNumber(1), BoundaryNumber(1, False)),
                          (BoundaryNumber(1, False), BoundaryNumber(1))])
def test_invert(a, b):
    assert ~a == b
