"""
Utility classes and functions
"""
from __future__ import annotations
from typing import Callable, List, Tuple
from collections.abc import Iterable
from dataclasses import dataclass
from operator import le, lt, ge, gt
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
        if self.lb_closed and self.ub_closed:
            assert self.lb <= self.ub
        else:
            assert self.lb < self.ub
        self.lb_closed = self.lb_closed and self.is_lb_finite()
        self.ub_closed = self.ub_closed and self.is_ub_finite()
        self.size = self.ub - self.lb

    def is_lb_finite(self) -> bool:
        return not np.isneginf(self.lb)

    def is_ub_finite(self) -> bool:
        return not np.isposinf(self.ub)

    def is_finite(self) -> bool:
        return self.is_lb_finite() and self.is_ub_finite()

    def __getitem__(self, index: int) -> float:
        assert index == 0 or index == 1
        return self.lb if index == 0 else self.ub

    def __repr__(self) -> str:
        lb_bracket = "[" if self.lb_closed else "("
        ub_bracket = "]" if self.ub_closed else ")"
        return f"{lb_bracket}{self.lb}, {self.ub}{ub_bracket}"


@dataclass
class IntervalFunction:
    fun: Callable
    invl: Interval

    def __call__(self, data: Iterable, order: int) -> np.ndarray:
        lb_opt = ge if self.invl.lb_closed or order < 0 else gt
        ub_opt = le if self.invl.ub_closed or order < 0 else lt
        data = np.asarray(data)
        if data.ndim == 2 and order >= 0:
            data = data[-1]
        assert lb_opt(data, self.invl.lb).all()
        assert ub_opt(data, self.invl.ub).all()
        return self.fun(data, order)


@dataclass
class SplineSpecs:
    knots: Iterable
    degree: int
    l_linx: bool = False
    r_linx: bool = False

    def __post_init__(self):
        assert isinstance(self.knots, Iterable)
        assert isinstance(self.degree, int) and self.degree >= 0
        assert isinstance(self.l_linx, bool)
        assert isinstance(self.r_linx, bool)
        assert len(self.knots) >= 2
        self.knots = np.unique(self.knots)

    def reset_degree(self, degree: int) -> SplineSpecs:
        return SplineSpecs(
            knots=self.knots,
            degree=degree,
            l_linx=self.l_linx,
            r_linx=self.r_linx
        )

    @property
    def num_spline_bases(self) -> int:
        return len(self.knots) + self.degree - 1


def process_fun_input(data: Iterable, order: int) -> Tuple[np.ndarray, int]:
    data = np.asarray(data)
    assert isinstance(order, int)
    if order < 0:
        assert data.ndim == 2
    if data.ndim == 2:
        assert (data[0] <= data[1]).all()
    return data, order


def taylor_term(x: np.ndarray, order: int) -> np.ndarray:
    assert isinstance(order, int)
    assert order >= 0
    if order == 0:
        result = np.ones(x.size)
    else:
        result = x**order/np.math.factorial(order)
    return result


def ind_fun(data: Iterable, order: int, invl: Interval) -> np.ndarray:
    data, order = process_fun_input(data, order)
    info = data if data.ndim == 1 else data[-1]
    if order == 0:
        lb_opt = ge if invl.lb_closed else gt
        ub_opt = le if invl.ub_closed else lt
        sub_index = lb_opt(info, invl.lb) & ub_opt(info, invl.ub)
        result = sub_index.astype(float)
    elif order > 0:
        result = np.zeros(info.size)
    else:
        one_fun = con_fun_factory(c=1.0)
        zero_fun = con_fun_factory(c=0.0)

        funs = [IntervalFunction(one_fun, invl)]
        if invl.is_lb_finite():
            fun_invl = Interval(lb=-np.inf,
                                ub=invl.lb,
                                ub_closed=not invl.lb_closed)
            funs.insert(0, IntervalFunction(zero_fun, fun_invl))
        if invl.is_ub_finite():
            fun_invl = Interval(lb=invl.ub,
                                ub=np.inf,
                                lb_closed=not invl.ub_closed)
            funs.append(IntervalFunction(zero_fun, fun_invl))
        result = combine_invl_funs(funs)(data, order)
    return result


def lin_fun(data: Iterable, order: int,
            z: float, fz: float, gz: float) -> np.ndarray:
    data, order = process_fun_input(data, order)
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
    points = np.linspace(invl.lb, invl.ub, len(weights))
    return lagrange(points, weights)(data)


def con_fun_factory(c: float = 1.0) -> Callable:
    def con_fun(data: np.ndarray, order: int) -> np.ndarray:
        data, order = process_fun_input(data, order)
        result = np.zeros(data.shape[-1])
        if c != 0.0:
            if order == 0:
                result.fill(c)
            elif order < 0:
                result = c*taylor_term(data[1] - data[0], -order)
        return result
    return con_fun


def combine_two_invl_funs(l_fun: IntervalFunction,
                          r_fun: IntervalFunction) -> IntervalFunction:
    assert l_fun.invl.ub == r_fun.invl.lb
    assert l_fun.invl.ub_closed ^ r_fun.invl.lb_closed

    l_opt = le if l_fun.invl.ub_closed else lt
    r_opt = ge if r_fun.invl.lb_closed else gt

    def fun(data: Iterable, order: int) -> np.ndarray:
        data, order = process_fun_input(data, order)
        x1 = data if data.ndim == 1 else data[-1]
        x1_l_index = l_opt(x1, l_fun.invl.ub)
        x1_r_index = r_opt(x1, r_fun.invl.lb)
        val = np.zeros(data.shape[-1])
        if order >= 0:
            val[x1_l_index] = l_fun(data[..., x1_l_index], order)
            val[x1_r_index] = r_fun(data[..., x1_r_index], order)
        else:
            x0 = data[0]
            x0_l_index = l_opt(x0, l_fun.invl.ub)
            x0_r_index = r_opt(x0, r_fun.invl.lb)
            val[x1_l_index] = l_fun(data[..., x1_l_index], order)
            val[x0_r_index] = r_fun(data[..., x0_r_index], order)
            rest_index = x0_l_index & x1_r_index
            l_data = data[..., rest_index].copy()
            r_data = data[..., rest_index].copy()
            l_data[1] = l_fun.invl.ub
            r_data[0] = r_fun.invl.lb
            for i in range(order + 1, 0):
                val[rest_index] += l_fun(l_data, i) * taylor_term(r_data[1] - r_data[0], i - order)
            val[rest_index] += l_fun(l_data, order) + r_fun(r_data, order)
        return val
    invl = Interval(lb=l_fun.invl.lb,
                    ub=r_fun.invl.ub,
                    lb_closed=l_fun.invl.lb_closed,
                    ub_closed=r_fun.invl.ub_closed)
    return IntervalFunction(fun, invl)


def combine_invl_funs(funs: List[IntervalFunction]) -> IntervalFunction:
    fun = funs[0]
    for i in range(1, len(funs)):
        fun = combine_two_invl_funs(fun, funs[i])

    return fun
