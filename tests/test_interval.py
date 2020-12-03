"""
Test Interval Module
"""
import pytest
import numpy as np
from xspline.interval import Interval


def test_default():
    invl = Interval()
    assert np.isneginf(invl.lb)
    assert np.isposinf(invl.ub)
    assert not invl.lb_closed
    assert not invl.ub_closed


@pytest.mark.parametrize(("lb", "ub", "lb_closed", "ub_closed"),
                         [(1.0, 0.0, False, False),
                          (1.0, 1.0, False, True),
                          (1.0, 1.0, True, False),
                          (-np.inf, -np.inf, True, True),
                          (np.inf, np.inf, True, True)])
def test_check_input(lb, ub, lb_closed, ub_closed):
    with pytest.raises(AssertionError):
        Interval(lb, ub, lb_closed, ub_closed)


@pytest.mark.parametrize(("lb", "ub", "size"),
                         [(0.0, 1.0, 1.0),
                          (-np.inf, 0.0, np.inf),
                          (0.0, np.inf, np.inf)])
def test_size(lb, ub, size):
    invl = Interval(lb, ub)
    assert invl.size == size


def test_eq_check():
    with pytest.raises(ValueError):
        Interval(0.0, 1.0) == 1.0


@pytest.mark.parametrize(("invl1", "invl2", "are_equal"),
                         [(Interval(0.0, 1.0), Interval(0.0, 1.0), True),
                          (Interval(0.0, 1.0, False), Interval(0.0, 1.0), False)])
def test_eq(invl1, invl2, are_equal):
    assert (invl1 == invl2) == are_equal


@pytest.mark.parametrize(("linvl", "rinvl"),
                         [(Interval(0.0, 1.0), Interval(1.5, 2.0)),
                          (Interval(0.0, 1.0), Interval(1.0, 2.0)),
                          (Interval(0.0, 1.0, ub_closed=False), Interval(1.0, 2.0, lb_closed=False))])
def test_check_add(linvl, rinvl):
    with pytest.raises(AssertionError):
        linvl + rinvl


def test_add():
    linvl = Interval(0.0, 1.0, True, False)
    rinvl = Interval(1.0, 2.0, True, False)
    assert linvl + rinvl == Interval(0.0, 2.0, True, False)


@pytest.mark.parametrize(("linvl", "rinvl"),
                         [(Interval(0.0, 1.0), Interval(1.5, 2.0)),
                          (Interval(0.0, 1.0, ub_closed=False), Interval(1.0, 2.0, lb_closed=False))])
def test_check_or(linvl, rinvl):
    with pytest.raises(AssertionError):
        linvl | rinvl


def test_or():
    linvl = Interval(0.0, 1.5, True, False)
    rinvl = Interval(1.0, 2.0, True, False)
    assert linvl | rinvl == Interval(0.0, 2.0, True, False)


@pytest.mark.parametrize(("linvl", "rinvl"),
                         [(Interval(0.0, 1.0), Interval(1.5, 2.0)),
                          (Interval(0.0, 1.0, ub_closed=False), Interval(1.0, 2.0, lb_closed=False))])
def test_check_and(linvl, rinvl):
    with pytest.raises(AssertionError):
        linvl & rinvl


def test_and():
    linvl = Interval(0.0, 1.5, True, False)
    rinvl = Interval(1.0, 2.0, True, False)
    assert linvl & rinvl == Interval(1.0, 1.5, True, False)


@pytest.mark.parametrize(("invl", "invert"),
                         [(Interval(-np.inf, np.inf), (None, None)),
                          (Interval(-np.inf, 0.0), (None, Interval(0.0, np.inf, False, False))),
                          (Interval(0.0, np.inf), (Interval(-np.inf, 0.0, False, False), None)),
                          (Interval(0.0, 1.0), (Interval(-np.inf, 0.0, False, False),
                                                Interval(1.0, np.inf, False, False)))])
def test_invert(invl, invert):
    assert ~invl == invert


def test_get_item():
    invl = Interval(0.0, 1.0)
    assert invl[0] == 0.0
    assert invl[1] == 1.0
