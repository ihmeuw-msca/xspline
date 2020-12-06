"""
Spline Basis Module
"""
from dataclasses import dataclass, field
from operator import xor
from typing import Iterable, List, Tuple, Union

import numpy as np

from xspline.fullfun import FullFunction
from xspline.funs import IndicatorFunction, PolyFunction
from xspline.funutils import check_fun_input, check_number
from xspline.interval import Interval


class SplineSpecs:
    def __init__(self, knots: Iterable, degree: int, index: int = None):
        self._knots = self._check_knots(knots)
        self._degree = self._check_degree(degree)
        self._num_bases = self._get_num_bases()
        self._index = self._check_index(index)
        self._domain = self._get_domain()
        self._support = self._get_support()

    def _check_knots(self, knots: Iterable) -> np.ndarray:
        knots = np.unique(knots)
        if knots.size < 2:
            raise ValueError("Need at least two unique knots.")
        if not (np.issubdtype(knots.dtype, int) or
                np.issubdtype(knots.dtype, float)):
            raise ValueError("Values in `knots` have to be integer or float.")
        return knots

    def _check_degree(self, degree: int) -> int:
        return check_number(degree, int, Interval(0, np.inf))

    def _check_index(self, index: Union[None, int]) -> Union[None, int]:
        if index is not None:
            index = check_number(index, int, Interval(0, self.num_bases - 1))
        return index

    @property
    def knots(self) -> np.ndarray:
        return self._knots

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def index(self) -> int:
        return self._index

    @property
    def num_bases(self) -> int:
        return self._num_bases

    @property
    def domain(self) -> Interval:
        return self._domain

    @property
    def support(self) -> Interval:
        return self._support

    @knots.setter
    def knots(self, knots: Iterable):
        self._knots = self._check_knots(knots)
        self._num_bases = self._get_num_bases()
        self._domain = self._get_domain()
        self._support = self._get_support()

    @degree.setter
    def degree(self, degree: int):
        self._degree = self._check_degree(degree)
        self._num_bases = self._get_num_bases()
        self._domain = self._get_domain()
        self._support = self._get_support()

    @index.setter
    def index(self, index: int):
        self._index = self._check_index(index)
        self._domain = self._get_domain()
        self._support = self._get_support()

    def _get_boundary_knots_indices(self) -> Tuple[int, int]:
        lb_index = max(self.index - self.degree, 0)
        ub_index = min(self.index + 1, self.knots.size - 1)
        return lb_index, ub_index

    def _get_num_bases(self) -> int:
        return self.knots.size + self.degree - 1

    def _get_domain(self) -> Interval:
        if self.index is None:
            domain = None
        else:
            knots = self.knots.copy()
            lb_index, ub_index = self._get_boundary_knots_indices()
            domain = Interval((knots[lb_index], True),
                              (knots[ub_index], False))
        return domain

    def _get_support(self) -> Interval:
        if self.index is None:
            support = None
        else:
            knots = self.knots.copy().astype(float)
            knots[0] = -np.inf
            knots[-1] = np.inf
            lb_index, ub_index = self._get_boundary_knots_indices()
            support = Interval((knots[lb_index], True),
                               (knots[ub_index], False))
        return support

    def copy(self, with_index: bool = False) -> "SplineSpecs":
        knots = self._knots.copy()
        degree = self._degree
        index = self._index if with_index else None
        return SplineSpecs(knots, degree, index)

    def __copy__(self, with_index: bool = False) -> "SplineSpecs":
        return self.copy(with_index=with_index)

    def __repr__(self) -> str:
        if self.index is None:
            return f"Spline(knots={self.knots}, degree={self.degree})"
        else:
            return (f"Spline(knots={self.knots}, "
                    f"degree={self.degree}, "
                    f"index={self.index}, "
                    f"domain={self.domain}, "
                    f"support={self.support})")


class SplineBasis(FullFunction):
    """Basic building block for splines.
    """

    def __init__(self,
                 specs: SplineSpecs,
                 links: List["SplineBasis"] = None,
                 **kwargs):
        if specs.index is None:
            raise ValueError("Please set `specs.index` first.")
        self.specs = specs
        self.links = [None, None] if links is None else links
        kwargs.update({"domain": self.specs.domain, "support": self.specs.support})
        super().__init__(**kwargs)

    def link_basis(self, basis: "SplineBasis"):
        assert isinstance(basis, SplineBasis)
        assert basis.specs.degree == self.specs.degree - 1
        assert np.allclose(basis.specs.knots, self.specs.knots)
        self.links[basis.specs.index - self.specs.index + 1] = basis

    def link_bases(self, bases: List["SplineBasis"]):
        assert len(bases) <= 2
        for basis in bases:
            self.link_basis(basis)

    def is_linked(self) -> bool:
        edges = [self.specs.index == 0,
                 self.specs.index == self.specs.num_bases - 1]
        links = [basis is not None for basis in self.links]
        return self.specs.degree == 0 or all([xor(*p) for p in zip(edges, links)])

    def __call__(self, data: Iterable, order: int = 0) -> np.ndarray:
        assert self.is_linked()
        data, order = check_fun_input(data, order)
        if self.specs.degree == 0:
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
        return self.specs.__str__()


def get_spline_bases(specs: SplineSpecs) -> List[List[SplineBasis]]:
    specs = specs.copy()
    bases = []
    # create bases
    for degree in range(specs.degree + 1):
        sub_specs = SplineSpecs(specs.knots, degree)
        bases.append([
            SplineBasis(SplineSpecs(specs.knots, degree, index))
            for index in range(sub_specs.num_bases)
        ])
    # link bases
    for degree in range(1, specs.degree + 1):
        for i, basis in enumerate(bases[degree]):
            indices = set([
                max(0, i - 1),
                min(i, len(bases[degree - 1]) - 1)
            ])
            basis.link_bases([bases[degree - 1][j] for j in indices])
    return bases
