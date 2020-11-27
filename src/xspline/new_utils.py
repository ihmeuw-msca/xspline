"""
Utility classes and functions
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from numbers import Number
from operator import ge, gt, le, lt
from typing import Callable, List, Tuple

import numpy as np
from scipy.interpolate import lagrange


@dataclass
class Interval:
    lb: float
    ub: float
    lb_closed: bool = True
    ub_closed: bool = True

    def __post_init__(self):
        assert isinstance(self.lb_closed, bool)
        assert isinstance(self.ub_closed, bool)
        self.lb = -np.inf if self.lb == "inf" else self.lb
        self.ub = np.inf if self.ub == "inf" else self.ub
        self.lb_closed = self.lb_closed and self.is_lb_finite()
        self.ub_closed = self.ub_closed and self.is_ub_finite()
        self.size = self.ub - self.lb
        if self.lb_closed and self.ub_closed:
            assert self.lb <= self.ub
        else:
            assert self.lb < self.ub

    def is_lb_finite(self) -> bool:
        return not np.isneginf(self.lb)

    def is_ub_finite(self) -> bool:
        return not np.isposinf(self.ub)

    def is_finite(self) -> bool:
        return self.is_lb_finite() and self.is_ub_finite()

    def __getitem__(self, index: int) -> float:
        assert index in [0, 1]
        return self.lb if index == 0 else self.ub

    def __repr__(self) -> str:
        lb_bracket = "[" if self.lb_closed else "("
        ub_bracket = "]" if self.ub_closed else ")"
        return f"{lb_bracket}{self.lb}, {self.ub}{ub_bracket}"


def check_integer(i: int,
                  lb: Number = -np.inf,
                  ub: Number = np.inf) -> int:
    assert isinstance(i, int)
    assert lb <= i < ub
    return i


def get_num_bases(knots: np.ndarray, degree: int) -> int:
    return len(knots) + degree - 1


def check_fun_input(data: Iterable, order: int) -> Tuple[np.ndarray, int]:
    data = np.asarray(data)
    assert isinstance(order, int)
    data = data[None, :] if data.ndim == 1 else data
    if order < 0:
        assert len(data) == 2
    if len(data) == 2:
        assert (data[0] <= data[1]).all()
    return data, order


def taylor_term(x: np.ndarray, order: int) -> np.ndarray:
    assert isinstance(order, int)
    assert order >= 0
    return x**order/np.math.factorial(order)


def ind_fun(data: Iterable, order: int, invl: Interval) -> np.ndarray:
    data, order = check_fun_input(data, order)
    fun0 = partial(con_fun, c=0.0)
    fun1 = partial(con_fun, c=1.0)
    funs = deque([fun1])
    sups = deque([invl])
    if invl.is_lb_finite():
        funs.appendleft(fun0)
        sups.appendleft(Interval("inf", invl.lb, ub_closed=not invl.lb_closed))
    if invl.is_ub_finite():
        funs.append(fun0)
        sups.append(Interval(invl.ub, "inf", lb_closed=not invl.ub_closed))
    result = combine_invl_funs(funs, sups)(data, order)
    return result


def lin_fun(data: Iterable, order: int,
            z: float, fz: float, gz: float) -> np.ndarray:
    data, order = check_fun_input(data, order)
    info = data if data.ndim == 1 else data[-1]
    if order == 0:
        result = fz + gz*(info - z)
    elif order > 0:
        result = np.zeros(info.size)
        if order == 1:
            result.fill(gz)
    else:
        dx = data[1] - data[0]
        fx = fz + gz*(data[0] - z)
        gx = gz
        result = gx*taylor_term(dx, -order + 1) + fx*taylor_term(dx, -order)
    return result


def lag_fun(data: Iterable, weights: Iterable, invl: Interval) -> np.ndarray:
    assert invl.is_finite()
    assert len(weights) >= 2
    data = np.asarray(data)
    data = data[-1] if data.ndim == 2 else data
    points = np.linspace(invl.lb, invl.ub, len(weights))
    return lagrange(points, weights)(data)


def con_fun(data: np.ndarray, order: int, c: float) -> np.ndarray:
    data, order = check_fun_input(data, order)
    result = np.zeros(data.shape[-1])
    if c != 0.0:
        if order == 0:
            result.fill(c)
        elif order < 0:
            result = c*taylor_term(data[1] - data[0], -order)
    return result


def combine_two_invl_funs(l_fun: Callable,
                          r_fun: Callable,
                          l_sup: Interval,
                          r_sup: Interval) -> Tuple[Callable, Interval]:
    assert l_sup.ub == r_sup.lb
    assert l_sup.ub_closed ^ r_sup.lb_closed

    l_opt = le if l_sup.ub_closed else lt
    r_opt = ge if r_sup.lb_closed else gt

    def fun(data: Iterable, order: int) -> np.ndarray:
        data, order = check_fun_input(data, order)
        l_logi = np.vstack([l_opt(d, l_sup.ub) for d in data])
        r_logi = np.vstack([r_opt(d, r_sup.lb) for d in data])
        val = np.zeros(data.shape[-1])
        if order >= 0:
            val[l_logi[-1]] = l_fun(data[:, l_logi[-1]], order)
            val[r_logi[-1]] = r_fun(data[:, r_logi[-1]], order)
        else:
            val[l_logi[1]] = l_fun(data[:, l_logi[1]], order)
            val[r_logi[0]] = l_fun(data[:, r_logi[0]], order)
            logi = l_logi[0] & r_logi[1]
            l_data = np.vstack([data[0, logi], np.repeat(l_sup.ub, logi.sum())])
            r_data = np.vstack([np.repeat(r_sup.lb, logi.sum()), data[1, logi]])
            for i in range(order + 1, 0):
                val[logi] += l_fun(l_data, i)*taylor_term(
                    r_data[1] - r_data[0], i - order
                )
            val[logi] += l_fun(l_data, order) + r_fun(r_data, order)
        return val
    invl = Interval(lb=l_sup.lb,
                    ub=r_sup.ub,
                    lb_closed=l_sup.lb_closed,
                    ub_closed=r_sup.ub_closed)
    return fun, invl


def combine_invl_funs(funs: List[Callable], sups: List[Interval]) -> Callable:
    assert len(funs) == len(sups)
    fun = funs[0]
    sup = sups[0]
    for i in range(1, len(funs)):
        fun, sup = combine_two_invl_funs(fun, funs[i],
                                         sup, sups[i])
    return fun
