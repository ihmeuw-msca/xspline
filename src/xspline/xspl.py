from typing import Optional

from numpy.typing import NDArray

from xspline.bspl import clear_bspl_cache, get_bspl_funs
from xspline.poly import get_poly_fun
from xspline.xfunction import BasisXFunction


class XSpline(BasisXFunction):

    def __init__(self,
                 knots: tuple[float, ...],
                 degree: int,
                 ldegree: Optional[int] = None,
                 rdegree: Optional[int] = None,
                 coefs: Optional[NDArray] = None) -> None:
        # validate inputs
        knots, degree = tuple(sorted(map(float, knots))), int(degree)
        if len(set(knots)) < 2:
            raise ValueError("please provide at least provide 2 distinct knots")
        if degree < 0:
            raise ValueError("degree must be nonnegative")
        ldegree = min(int(degree if ldegree is None else ldegree), degree)
        rdegree = min(int(degree if rdegree is None else rdegree), degree)

        # create basis functions
        mfuns = get_bspl_funs(knots, degree)
        lfuns = tuple(get_poly_fun(fun, knots[0], ldegree) for fun in mfuns)
        rfuns = tuple(get_poly_fun(fun, knots[-1], rdegree) for fun in mfuns)
        funs = tuple(
            lfun.append(mfun, (knots[0], False)).append(rfun, (knots[-1], True))
            for lfun, mfun, rfun in zip(lfuns, mfuns, rfuns)
        )

        self.knots, self.degree = knots, degree
        self.ldegree, self.rdegree = ldegree, rdegree
        super().__init__(funs, coefs=coefs)

    def get_design_mat(self, x: NDArray,
                       order: int = 0, check_args: bool = True) -> NDArray:
        design_mat = super().get_design_mat(x, order, check_args)
        clear_bspl_cache()
        return design_mat
