"""
Spline Basis Module
"""
from __future__ import annotations
from typing import List, Union
from collections.abc import Iterable
from dataclasses import dataclass, field
from operator import xor
import numpy as np
from .new_utils import Interval, SplineSpecs
from .new_utils import ind_fun, lag_fun


@dataclass
class BasisLinks:
    """Links between bases
    """
    bases: List[SplineBasis] = field(default_factory=lambda: [None, None])

    def __post_init__(self):
        assert len(self.bases) == 2
        assert all([self.is_basis(basis) for basis in self.bases])

    def is_basis(self, basis: Union[SplineBasis, None]):
        return isinstance(basis, SplineBasis) or basis is None

    def is_empty(self) -> bool:
        return all([basis is None for basis in self.bases])

    def is_l_linked(self) -> bool:
        return self.bases[0] is not None

    def is_r_linked(self) -> bool:
        return self.bases[1] is not None

    def is_linked(self) -> List[bool]:
        return [self.is_l_linked(), self.is_r_linked()]

    def link_basis(self, basis: SplineBasis, index: int):
        assert self.is_basis(basis)
        self.bases[index] = basis


@dataclass
class SplineBasis:
    """Basic building block for splines.
    """
    specs: SplineSpecs
    index: int
    links: BasisLinks = field(default_factory=BasisLinks)

    def __post_init__(self):
        assert isinstance(self.index, int)
        assert 0 <= self.index < self.specs.num_spline_bases

        lb_index = max(self.index - self.specs.degree, 0)
        ub_index = min(self.index + 1, len(self.specs.knots) - 1)
        reach_lb = lb_index == 0
        reach_ub = ub_index == len(self.specs.knots) - 1

        self.domain = Interval(
            lb=self.specs.knots[lb_index],
            ub=self.specs.knots[ub_index],
            lb_closed=True, ub_closed=reach_ub
        )
        self.support = Interval(
            lb=-np.inf if reach_lb else self.domain.lb,
            ub=np.inf if reach_ub else self.domain.ub,
            lb_closed=not reach_lb, ub_closed=False
        )

        self.data = np.empty((2, 0))
        self.vals = {}

    def is_l_edge(self) -> bool:
        return self.index == 0

    def is_r_edge(self) -> bool:
        return self.index == self.specs.num_spline_bases - 1

    def is_edge(self) -> List[bool]:
        return [self.is_l_edge(), self.is_r_edge()]

    def link_basis(self, basis: SplineBasis):
        assert isinstance(basis, SplineBasis)
        assert basis.specs.degree == self.specs.degree - 1
        self.links.link_basis(basis, basis.index - self.index + 1)

    def link_bases(self, bases: List[SplineBasis]):
        assert len(bases) <= 2
        for basis in bases:
            self.link_basis(basis)

    def is_linked(self) -> bool:
        return self.specs.degree == 0 or all([
            xor(*pair)
            for pair in zip(self.is_edge(), self.links.is_linked())
        ])

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
                self.data[0] = self.specs.knots[0]
                self.data[1] = x
            assert (self.data[0] <= self.data[1]).all()

    def fun(self):
        if self.specs.degree == 0:
            self.vals[0] = ind_fun(self.data[1], 0, self.support)
        else:
            self.vals[0] = np.zeros(self.data.shape[1])
            if self.links.is_l_linked():
                basis = self.links.bases[0]
                self.vals[0] += basis(self.data[1], order=0)*lag_fun(self.data[1], [0, 1], basis.domain)
            if self.links.is_r_linked():
                basis = self.links.bases[1]
                self.vals[0] += basis(self.data[1], order=0)*lag_fun(self.data[1], [1, 0], basis.domain)

    def difun(self, order: int):
        if self.specs.degree == 0:
            self.vals[order] = ind_fun(self.data, order, self.support)
        else:
            self.vals[order] = np.zeros(self.data.shape[1])
            if self.specs.degree - order >= 0:
                if self.links.is_l_linked():
                    basis = self.links.bases[0]
                    self.vals[order] += (
                        basis(self.data, order=order)*lag_fun(self.data[1], [0, 1], basis.domain) +
                        basis(self.data, order=order - 1)*order/basis.domain.size
                    )
                if self.links.is_r_linked():
                    basis = self.links.bases[1]
                    self.vals[order] += (
                        basis(self.data, order=order)*lag_fun(self.data[1], [1, 0], basis.domain) -
                        basis(self.data, order=order - 1)*order/basis.domain.size
                    )

    def __call__(self, data: np.ndarray, order: int = 0) -> np.ndarray:
        assert isinstance(order, int)
        assert self.is_linked()
        self.attach_data(data)

        if order not in self.vals:
            if order == 0:
                self.fun()
            else:
                self.difun(order=order)

        return self.vals[order]

    def __repr__(self) -> str:
        return f"SplineBasis(degree={self.specs.degree}, index={self.index}, domain={self.domain})"
