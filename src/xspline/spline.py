"""
Spline Module
"""
from typing import Iterable

import numpy as np

from xspline.basis import SplineBasis, SplineSpecs


class XSpline:
    def __init__(self,
                 knots: Iterable,
                 degree: int,
                 l_linx: bool = False,
                 r_linx: bool = False):
        self.specs = SplineSpecs(knots, degree)
        self.l_linx = l_linx
        self.r_linx = r_linx
        self.bases = []
        for d in range(self.specs.degree + 1):
            specs = SplineSpecs(self.specs.knots, d, i)
            self.bases.append([
                SplineBasis(specs)
                for i in range(specs.num_bases)
            ])

        for d in range(1, self.specs.degree + 1):
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
