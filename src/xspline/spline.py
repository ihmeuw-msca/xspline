"""
Spline Module
"""
from typing import Iterable, List

import numpy as np

from xspline.basis import SplineBasis, SplineSpecs, get_spline_bases
from xspline.fullfun import FullFunction
from xspline.funs import PolyFunction
from xspline.funutils import check_number
from xspline.interval import Interval


class XSpline:
    def __init__(self,
                 knots: Iterable,
                 degree: int,
                 lxorder: int = None,
                 rxorder: int = None):
        self.specs = SplineSpecs(knots, degree)
        self.bases = get_spline_bases(self.specs)

        lxorder = self.specs.degree if lxorder is None else \
            check_number(lxorder, int, Interval(0, self.specs.degree))
        rxorder = self.specs.degree if rxorder is None else \
            check_number(rxorder, int, Interval(0, self.specs.degree))
        self.xorders = (lxorder, rxorder)

        self.domain = Interval(self.specs.knots[0], (self.specs.knots[-1], False))
        self.funs = self._get_spline_funs()

    def _get_spline_funs(self) -> List[FullFunction]:
        xdomains = ~self.domain
        funs = []
        for basis in self.bases[-1]:
            for i, xorder in enumerate(self.xorders):
                if basis.domain[i] == self.domain[i]:
                    fun_vals = [basis([self.domain[i]], order)[0]
                                for order in range(xorder + 1)]
                    xfun = PolyFunction.from_taylor(self.domain[i], fun_vals, domain=xdomains[i])
                    basis = xfun + basis if i == 0 else basis + xfun
            funs.append(basis)
        return funs

    def design_mat(self, data: np.ndarray, order: int = 0) -> np.ndarray:
        return np.hstack([
            fun(data, order)[:, None]
            for fun in self.funs
        ])

    def __repr__(self) -> str:
        return self.specs.__str__() + f" + xorders={self.xorders}"
