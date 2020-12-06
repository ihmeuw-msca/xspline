"""
Spline Basis Module
"""
from dataclasses import dataclass, field
from operator import xor
from typing import Iterable, List

import numpy as np

from xspline.fullfun import FullFunction
from xspline.funs import IndicatorFunction, PolyFunction
from xspline.funutils import check_fun_input, check_number
from xspline.interval import Interval


class SplineBasis(FullFunction):
    """Basic building block for splines.
    """

    def __init__(self,
                 knots: Iterable,
                 degree: int,
                 index: int,
                 links: List["SplineBasis"] = None, **kwargs):
        self.knots = np.unique(knots)
        self.degree = check_number(degree, int, Interval(0, np.inf))
        self.index = check_number(index, int, Interval(0, len(self.knots) + self.degree - 1))
        self.links = [None, None] if links is None else links

        lb_index = max(self.index - self.degree, 0)
        ub_index = min(self.index + 1, len(self.knots) - 1)
        reach_lb = lb_index == 0
        reach_ub = ub_index == len(self.knots) - 1

        domain = Interval(self.knots[lb_index], (self.knots[ub_index], False))
        support = Interval(-np.inf if reach_lb else domain.lb,
                           np.inf if reach_ub else domain.ub)
        super().__init__(domain=domain, support=support, **kwargs)

    def link_basis(self, basis: "SplineBasis"):
        assert isinstance(basis, SplineBasis)
        assert basis.degree == self.degree - 1
        assert np.allclose(basis.knots, self.knots)
        self.links[basis.index - self.index + 1] = basis

    def link_bases(self, bases: List["SplineBasis"]):
        assert len(bases) <= 2
        for basis in bases:
            self.link_basis(basis)

    def is_linked(self) -> bool:
        edges = [self.index == 0,
                 self.index == len(self.knots) + self.degree - 2]
        links = [basis is not None for basis in self.links]
        return self.degree == 0 or all([xor(*p) for p in zip(edges, links)])

    def __call__(self, data: Iterable, order: int = 0) -> np.ndarray:
        assert self.is_linked()
        data, order = check_fun_input(data, order)
        if self.degree == 0:
            val = IndicatorFunction(domain=self.support)(data, order)
        else:
            val = np.zeros(data.shape[-1])
            for i, basis in enumerate(self.links):
                if basis is None:
                    continue
                lag_fun = PolyFunction.from_lagrange(list(basis.domain), [i, 1 - i])
                val += basis(data, order)*lag_fun(data)
                if order != 0:
                    val += order*basis(data, order - 1)*lag_fun(data, 1)
        return val

    def __repr__(self) -> str:
        return f"SplineBasis(degree={self.degree}, index={self.index}, domain={self.domain})"


class XSpline:
    def __init__(self, knots: Iterable, degree: int,
                 l_linx: bool = False, r_linx: bool = False):
        self.knots = np.unique(knots)
        self.degree = check_number(degree, int, Interval(0, np.inf))
        self.num_bases = len(self.knots) + self.degree - 1
        self.l_linx = l_linx
        self.r_linx = r_linx
        self.bases = []
        for d in range(self.degree + 1):
            self.bases.append([
                SplineBasis(self.knots, d, i)
                for i in range(len(self.knots) + d - 1)
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

    def design_mat(self, data: np.ndarray, order: int = 0) -> np.ndarray:
        data = np.asarray(data)
        if data.ndim == 1 and order < 0:
            data = np.vstack([np.repeat(data.min(), data.size), data])
        return np.hstack([
            fun(data, order)[:, None]
            for fun in self.bases[-1]
        ])
