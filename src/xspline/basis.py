"""
Spline Basis Module
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from functools import partial
from operator import xor
from typing import Iterable, List

import numpy as np

from xspline.interval import Interval
from xspline.fullfun import FullFunction
from xspline.funutils import check_fun_input, check_number, lag_fun
from xspline.funs import IndicatorFunction


@dataclass
class SplineBasis(FullFunction):
    """Basic building block for splines.
    """
    knots: Iterable
    degree: int
    index: int
    links: List[SplineBasis] = field(default_factory=lambda: [None, None])

    def __post_init__(self):
        self.knots = np.unique(self.knots)
        self.degree = check_number(self.degree, int, Interval(0, np.inf))
        self.index = check_number(self.index, int, Interval(0, len(self.knots) + self.degree - 1))

        lb_index = max(self.index - self.degree, 0)
        ub_index = min(self.index + 1, len(self.knots) - 1)
        reach_lb = lb_index == 0
        reach_ub = ub_index == len(self.knots) - 1

        self.domain = Interval(
            self.knots[lb_index],
            self.knots[ub_index],
            lb_closed=True, ub_closed=reach_ub
        )
        self.support = Interval(
            -np.inf if reach_lb else self.domain.lb,
            np.inf if reach_ub else self.domain.ub,
            lb_closed=not reach_lb, ub_closed=False
        )
        self.fun = IndicatorFunction(domain=self.support)

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
                 self.index == len(self.knots) + self.degree - 2]
        links = [basis is not None for basis in self.links]
        return self.degree == 0 or all([xor(*p) for p in zip(edges, links)])

    def __call__(self, data: Iterable, order: int = 0) -> np.ndarray:
        assert self.is_linked()
        if self.degree == 0:
            val = self.fun(data, order)
        else:
            val = np.zeros(data.shape[-1])
            for i, basis in enumerate(self.links):
                if basis is None:
                    continue
                val += basis(data, order)*lag_fun(data, [i, 1 - i], basis.domain)
                if order != 0:
                    val += (1 - 2*i)*basis(data, order - 1)*order/basis.domain.size
        return val

    def __repr__(self) -> str:
        return f"SplineBasis(degree={self.degree}, index={self.index}, domain={self.domain})"


# class XSpline:
#     def __init__(self, knots: Iterable, degree: int,
#                  l_linx: bool = False, r_linx: bool = False):
#         self.knots = np.unique(knots)
#         self.degree = check_integer(degree, lb=0)
#         self.num_bases = get_num_bases(self.knots, self.degree)
#         self.l_linx = l_linx
#         self.r_linx = r_linx
#         self.bases = []
#         for d in range(self.degree + 1):
#             self.bases.append([
#                 SplineBasis(self.knots, d, i)
#                 for i in range(get_num_bases(self.knots, d))
#             ])

#         for d in range(1, self.degree + 1):
#             bases = self.bases[d]
#             bases_prev = self.bases[d - 1]
#             for i, basis in enumerate(bases):
#                 indices = set([
#                     max(0, i - 1),
#                     min(i, len(bases_prev) - 1)
#                 ])
#                 basis.link_bases([bases_prev[j] for j in indices])

#         invl = Interval(lb=self.knots.min() if self.l_linx else "inf",
#                         ub=self.knots.max() if self.r_linx else "inf",
#                         lb_closed=True,
#                         ub_closed=True)
#         outer_sups = [
#             Interval("inf", invl.lb,
#                      lb_closed=False, ub_closed=False) if invl.is_lb_finite() else None,
#             Interval(invl.ub, "inf",
#                      lb_closed=False, ub_closed=False) if invl.is_ub_finite() else None
#         ]
#         self.basis_funs = []
#         for i in range(self.num_bases):
#             basis = self.bases[-1][i]
#             if not (self.l_linx or self.r_linx):
#                 self.basis_funs.append(basis)
#             else:
#                 funs = deque([basis])
#                 sups = deque([invl])
#                 for j, sup in enumerate(outer_sups):
#                     if sup is not None:
#                         outer_fun = partial(lin_fun,
#                                             z=invl[j],
#                                             fz=basis([invl[j]], order=0)[0],
#                                             gz=basis([invl[j]], order=1)[0])
#                         index = 0 if j == 0 else len(funs)
#                         funs.insert(index, outer_fun)
#                         sups.insert(index, sup)
#                 self.basis_funs.append(combine_invl_funs(funs, sups))

#     def design_mat(self, data: np.ndarray, order: int = 0) -> np.ndarray:
#         data = np.asarray(data)
#         if data.ndim == 1 and order < 0:
#             data = np.vstack([np.repeat(data.min(), data.size), data])
#         return np.hstack([
#             fun(data, order)[:, None]
#             for fun in self.basis_funs
#         ])
