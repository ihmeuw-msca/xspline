"""
Spline Basis Module
"""
from __future__ import annotations
from warnings import warn
from typing import List, Dict, Union
from dataclasses import dataclass, field
from operator import xor
import numpy as np
from .data import SplineSpecs
from .utils import linear_lf, linear_rf


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
        self.inner_knots = self.knots[self.l_linear:
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
        return len(self.inner_knots) + self.degree - 1


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
    
    def is_linked(self) -> List[bool]:
        return [basis is not None for basis in self.bases]
    
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
        self.domain = 
        self.data = None
        self.vals = {}
    
    def is_edge(self) -> List[bool]:
        return [self.index == 0, self.index == self.specs.num_spline_bases]
    
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
        return NotImplementedError()

    def dfun(self, order: int):
        return NotImplementedError()
    
    def ifun(self, order: int):
        return NotImplementedError()
        
    def __call__(self, data: np.ndarray, order: int = 0) -> np.ndarray:
        """Compute the spline base values.
        Args:
            data (np.ndarray): Independent variable.
            order (int, optional):
                Indicate differentiation or integration. When positive
                it is order of integration, when negative it is order of
                differentiation, and when zero, return function value.
                Defaults to 0.
        Returns:
            np.ndarray: Dependent variable.
        """
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
        return f"SplineBasis(degree={self.specs.degree}, index={self.index})"
