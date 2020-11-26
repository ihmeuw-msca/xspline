"""
Spline Basis Module
"""
from __future__ import annotations
from warnings import warn
from typing import List, Dict, Union, Iterable
from dataclasses import dataclass, field
from operator import xor, le, lt, ge, gt
import numpy as np
from scipy.interpolate import lagrange
from .utils import linear_lf, linear_rf


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

    def is_lb_finite(self) -> bool:
        return not np.isneginf(self.lb)
    
    def is_ub_finite(self) -> bool:
        return not np.isposinf(self.ub)

    def is_finite(self) -> bool:
        return self.is_lb_finite() and self.is_ub_finite()

    def indicator(self, data: Iterable) -> np.ndarray:
        data = np.asarray(data)
        lb_opt = ge if self.lb_closed else gt
        ub_opt = le if self.ub_closed else lt
        sub_index = lb_opt(data, self.lb) & ub_opt(data, self.ub)
        return sub_index.astype(float)

    def lagrange(self, data: Iterable, weights: np.ndarray) -> np.ndarray:
        assert self.is_finite()
        assert len(weights) >= 2
        data = np.asarray(data)
        lb_opt = ge if self.lb_closed else gt
        ub_opt = le if self.ub_closed else lt
        sub_index = lb_opt(data, self.lb) & ub_opt(data, self.ub)

        points = np.linspace(self.lb, self.ub, len(weights))
        poly = lagrange(points, weights)

        result = np.zeros(data.shape)
        result[sub_index] = poly(data[sub_index])
        return result

    def __repr__(self) -> str:
        lb_bracket = "[" if self.lb_closed else "("
        ub_bracket = "]" if self.ub_closed else ")"
        return f"{lb_bracket}{self.lb}, {self.ub}{ub_bracket}"

@dataclass
class SplineSpecs:
    knots: Iterable
    degree: int
    l_linear: bool = False
    r_linear: bool = False

    def __post_init__(self):
        assert isinstance(self.knots, Iterable)
        assert isinstance(self.degree, int) and self.degree >= 0
        assert isinstance(self.l_linear, bool)
        assert isinstance(self.r_linear, bool)
        assert len(self.knots) >= 2 + self.l_linear + self.r_linear

        self.knots = np.unique(self.knots)
        self.basis_knots = self.knots[self.l_linear:
                                      len(self.knots) - self.r_linear]
    
    def reset_degree(self, degree: int) -> SplineSpecs:
        return SplineSpecs(
            knots=self.knots,
            degree=degree,
            l_linear=self.l_linear,
            r_linear=self.r_linear
        )
    
    @property
    def num_spline_bases(self) -> int:
        return len(self.basis_knots) + self.degree - 1


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
        ub_index = min(self.index + 1, len(self.specs.basis_knots) - 1)

        self.domain = Interval(
            lb=self.specs.basis_knots[lb_index],
            ub=self.specs.basis_knots[ub_index],
            lb_closed=True, ub_closed=self.is_r_edge()
        )
        self.support = Interval(
            lb=-np.inf if self.is_l_edge() else self.domain.lb,
            ub= np.inf if self.is_r_edge() else self.domain.ub,
            lb_closed=not self.is_l_edge(), ub_closed=False
        )

        self.data = None
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
        return all([xor(*pair)
                    for pair in zip(self.is_edge(), self.links.is_linked())])
    
    def clear(self):
        self.data = None
        self.vals = {}
    
    def has_data(self, data: np.ndarray) -> bool:
        return (self.data is data) or (
            self.data.size == data.size and
            np.allclose(self.data, data)
        )

    def attach_data(self, data: np.ndarray) -> bool:
        data = np.asarray(data)
        if not self.has_data(data):
            self.clear()
            self.data = data

    def fun(self):
        if self.specs.degree == 0:
            self.vals[0] = self.support.indicator(self.data)
        else:
            self.vals[0] = np.zeros(self.data.shape)
            if self.links.is_l_linked():
                basis = self.links.bases[0]
                self.vals[0] += basis(self.data, order=0)*basis.domain.lagrange(self.data, [0, 1])
            if self.links.is_r_linked():
                basis = self.links.bases[1]
                self.vals[0] += basis(self.data, order=0)*basis.domain.lagrange(self.data, [1, 0])

    def dfun(self, order: int):
        return NotImplementedError()
    
    def ifun(self, order: int):
        return NotImplementedError()
        
    def __call__(self, data: np.ndarray, order: int = 0) -> np.ndarray:
        assert isinstance(order, int)
        assert self.is_linked()
        self.attach_data(data)

        if order not in self.vals:
            if order == 0:
                self.fun()
            elif order < 0:
                self.dfun(order=order)
            else:
                self.ifun(order=order)

        return self.vals[order]

    def __repr__(self) -> str:
        return f"SplineBasis(degree={self.specs.degree}, index={self.index}, domain={self.domain})"
