"""
Spline Module
"""
from typing import Iterable

import numpy as np

from xspline.basis import SplineBasis, SplineSpecs, get_spline_bases


class XSpline:
    def __init__(self,
                 knots: Iterable,
                 degree: int):
        self.specs = SplineSpecs(knots, degree)
        self.bases = get_spline_bases(self.specs)

    def design_mat(self, data: np.ndarray, order: int = 0) -> np.ndarray:
        data = np.asarray(data)
        if data.ndim == 1 and order < 0:
            data = np.vstack([np.repeat(data.min(), data.size), data])
        return np.hstack([
            fun(data, order)[:, None]
            for fun in self.bases[-1]
        ])

    def __repr__(self) -> str:
        return self.specs.__str__()
