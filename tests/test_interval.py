"""
Test Interval Module
"""
import pytest
import numpy as np
from xspline.interval import Interval


def test_default():
    invl = Interval()
    assert np.isneginf(invl.lb.val)
    assert np.isposinf(invl.ub.val)
    assert not invl.lb.cld
    assert not invl.ub.cld


@pytest.mark.parametrize(("lb", "ub"),
                         [((1.0, False), (0.0, False)),
                          ((1.0, False), (1.0, True)),
                          ((1.0, True), (1.0, False)),
                          ((-np.inf, True), (-np.inf, True)),
                          ((np.inf, True), (np.inf, True))])
def test_check_input(lb, ub):
    with pytest.raises(ValueError):
        Interval(lb, ub)


@ pytest.mark.parametrize(("lb", "ub", "size"),
                          [(0.0, 1.0, 1.0),
                           (-np.inf, 0.0, np.inf),
                           (0.0, np.inf, np.inf)])
def test_size(lb, ub, size):
    invl = Interval(lb, ub)
    assert invl.size == size


def test_eq_check():
    with pytest.raises(AssertionError):
        Interval(0.0, 1.0) == 1.0


@ pytest.mark.parametrize(("invl1", "invl2", "result"),
                          [(Interval(0.0, 1.0), Interval(0.0, 1.0), True),
                           (Interval((0.0, False), 1.0), Interval(0.0, 1.0), False)])
def test_eq(invl1, invl2, result):
    assert (invl1 == invl2) == result


@ pytest.mark.parametrize(("linvl", "rinvl"),
                          [(Interval(0.0, 1.0), Interval(1.5, 2.0)),
                           (Interval(0.0, 1.0), Interval(1.0, 2.0)),
                           (Interval(0.0, (1.0, False)), Interval((1.0, False), 2.0))])
def test_check_add(linvl, rinvl):
    with pytest.raises(AssertionError):
        linvl + rinvl


def test_add():
    linvl = Interval((0.0, True), (1.0, False))
    rinvl = Interval((1.0, True), (2.0, False))
    assert linvl + rinvl == Interval((0.0, True), (2.0, False))


@ pytest.mark.parametrize(("linvl", "rinvl"),
                          [(Interval(0.0, 1.0), Interval(1.5, 2.0)),
                           (Interval(0.0, (1.0, False)), Interval((1.0, False), 2.0))])
def test_check_or(linvl, rinvl):
    with pytest.raises(AssertionError):
        linvl | rinvl


def test_or():
    linvl = Interval((0.0, True), (1.5, False))
    rinvl = Interval((1.0, True), (2.0, False))
    assert linvl | rinvl == Interval((0.0, True), (2.0, False))


@ pytest.mark.parametrize(("linvl", "rinvl"),
                          [(Interval(0.0, 1.0), Interval(1.5, 2.0)),
                           (Interval(0.0, (1.0, False)), Interval((1.0, False), 2.0))])
def test_check_and(linvl, rinvl):
    with pytest.raises(AssertionError):
        linvl & rinvl


def test_and():
    linvl = Interval((0.0, True), (1.5, False))
    rinvl = Interval((1.0, True), (2.0, False))
    assert linvl & rinvl == Interval((1.0, True), (1.5, False))


@ pytest.mark.parametrize(("invl", "invert"),
                          [(Interval(-np.inf, np.inf), (None, None)),
                           (Interval(-np.inf, 0.0), (None, Interval((0.0, False), np.inf))),
                           (Interval(0.0, np.inf), (Interval(-np.inf, (0.0, False)), None)),
                           (Interval(0.0, 1.0), (Interval(-np.inf, (0.0, False)),
                                                 Interval((1.0, False), np.inf)))])
def test_invert(invl, invert):
    assert ~invl == invert


def test_get_item():
    invl = Interval(0.0, 1.0)
    assert invl[0] == 0.0
    assert invl[1] == 1.0


@ pytest.mark.parametrize("invl", [Interval(0.0, 1.0)])
@ pytest.mark.parametrize(("num", "isin"),
                          [(-1.0, False),
                           (0.0, True),
                           (0.5, True),
                           (1.0, True),
                           (1.5, False)])
def test_contains(invl, num, isin):
    assert (num in invl) == isin
