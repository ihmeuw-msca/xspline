"""
Spline Basis Module
"""
from __future__ import annotations
from typing import List, Iterable
from numbers import Number
from dataclasses import dataclass, field
from operator import xor
from functools import partial
import numpy as np
from .new_utils import Interval, IntervalFunction, SplineSpecs
from .new_utils import ind_fun, lag_fun, lin_fun, combine_invl_funs


def check_integer(i: int,
                  lb: Number = -np.inf,
                  ub: Number = np.inf) -> int:
    assert isinstance(i, int)
    assert lb <= i < ub
    return i


def get_num_bases(knots: np.ndarray, degree: int) -> int:
    return len(knots) + degree - 1


@dataclass
class SplineBasis:
    """Basic building block for splines.
    """
    knots: Iterable
    degree: int
    index: int
    links: List[SplineBasis] = field(default_factory=lambda: [None, None])

    def __post_init__(self):
        self.knots = np.unique(self.knots)
        self.degree = check_integer(self.degree, lb=0)
        self.num_bases = get_num_bases(self.knots, self.degree)
        self.index = check_integer(self.index, lb=0, ub=self.num_bases)

        lb_index = max(self.index - self.degree, 0)
        ub_index = min(self.index + 1, len(self.knots) - 1)
        reach_lb = lb_index == 0
        reach_ub = ub_index == len(self.knots) - 1

        self.domain = Interval(
            lb=self.knots[lb_index],
            ub=self.knots[ub_index],
            lb_closed=True, ub_closed=reach_ub
        )
        self.support = Interval(
            lb=-np.inf if reach_lb else self.domain.lb,
            ub=np.inf if reach_ub else self.domain.ub,
            lb_closed=not reach_lb, ub_closed=False
        )

        self.data = np.empty((2, 0))
        self.vals = {}

    def link_basis(self, basis: SplineBasis):
        assert isinstance(basis, SplineBasis)
        assert basis.degree == self.degree - 1
        assert np.allclose(basis.knots, self.knots)
        self.links[basis.index - self.index + 1] = basis

    def link_bases(self, bases: List[SplineBasis]):
        assert len(bases) <= 2
        for basis in bases:
            self.link_basis(basis)

    def is_linked(self) -> bool:
        edges = [self.index == 0,
                 self.index == self.num_bases - 1]
        links = [basis is not None for basis in self.links]
        return self.degree == 0 or all([xor(*p) for p in zip(edges, links)])

    def clear(self):
        self.data = np.empty((2, 0))
        self.vals = {}

    def has_data(self, x: np.ndarray = None) -> bool:
        if x is None:
            result = self.data.size > 0
        else:
            data = self.data if x.ndim == 2 else self.data[1]
            result = data.shape == x.shape and np.allclose(data, x)
        return result

    def attach_data(self, x: np.ndarray):
        x = np.asarray(x)
        if not self.has_data(x):
            self.clear()
            if x.ndim == 2:
                self.data = x
            else:
                self.data = np.empty((2, x.size))
                self.data[0] = x.min()
                self.data[1] = x
            assert (self.data[0] <= self.data[1]).all()

    def fun(self, order: int):
        if self.degree == 0:
            self.vals[order] = ind_fun(self.data, order, self.support)
        else:
            self.vals[order] = np.zeros(self.data.shape[1])
            for i, basis in enumerate(self.links):
                if basis is not None:
                    lag_val = lag_fun(self.data[1], [i, 1 - i], basis.domain)
                    self.vals[order] += basis(self.data, order=order)*lag_val
                    if order != 0:
                        self.vals[order] += np.sign(0.5 - i)*basis(self.data, order=order - 1)*order/basis.domain.size

    def __call__(self, data: np.ndarray, order: int = 0) -> np.ndarray:
        assert isinstance(order, int)
        assert self.is_linked()
        self.attach_data(data)

        if order not in self.vals:
            self.fun(order)

        return self.vals[order]

    def __repr__(self) -> str:
        return f"SplineBasis(degree={self.degree}, index={self.index}, domain={self.domain})"


class XSpline:
    def __init__(self, knots: Iterable, degree: int,
                 l_linx: bool = False, r_linx: bool = False):
        self.knots = np.unique(knots)
        self.degree = check_integer(degree, lb=0)
        self.num_bases = get_num_bases(self.knots, self.degree)
        self.l_linx = l_linx
        self.r_linx = r_linx
        self.bases = []
        for d in range(self.degree + 1):
            self.bases.append([
                SplineBasis(self.knots, d, i)
                for i in range(get_num_bases(self.knots, d))
            ])

        for d in range(1, self.degree + 1):
            bases = self.bases[d]
            bases_prev = self.bases[d - 1]
            for i, basis in enumerate(bases):
                indices = set([
                    max(0, i - 1),
                    min(i, len(bases_prev) - 1)
                ])
                basis.link_bases([bases_prev[j] for j in indices])

        invl = Interval(lb=self.knots[0] if self.l_linx else -np.inf,
                        ub=self.knots[-1] if self.r_linx else np.inf,
                        lb_closed=True,
                        ub_closed=True)
        outer_invls = [
            Interval(lb=-np.inf, ub=invl.lb,
                     lb_closed=False, ub_closed=False) if invl.is_lb_finite() else None,
            Interval(lb=invl.ub, ub=np.inf,
                     lb_closed=False, ub_closed=False) if invl.is_ub_finite() else None
        ]
        self.basis_funs = []
        for i in range(self.num_bases):
            basis = self.bases[-1][i]
            if not (self.l_linx or self.r_linx):
                self.basis_funs.append(basis)
            else:
                funs = [IntervalFunction(basis, invl)]
                for j, outer_invl in enumerate(outer_invls):
                    if outer_invl is not None:
                        outer_fun = partial(lin_fun,
                                            z=invl[j],
                                            fz=basis([invl[j]], order=0)[0],
                                            gz=basis([invl[j]], order=1)[0])
                        index = 0 if j == 0 else len(funs)
                        funs.insert(index, IntervalFunction(outer_fun, outer_invls[j]))
                self.basis_funs.append(combine_invl_funs(funs))

    def design_mat(self, data: np.ndarray, order: int = 0) -> np.ndarray:
        data = np.asarray(data)
        if data.ndim == 1 and order < 0:
            data = np.vstack([np.repeat(data.min(), data.size), data])
        return np.hstack([
            fun(data, order)[:, None]
            for fun in self.basis_funs
        ])
