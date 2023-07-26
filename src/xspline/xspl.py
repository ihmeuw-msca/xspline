from typing import Optional

from xspline.bspl import clear_bspl_cache, get_bspl_funs
from xspline.poly import get_poly_fun
from xspline.xfunction import BasisXFunction
from xspline.typing import NDArray


class XSpline(BasisXFunction):
    """Main class for xspline functions.

    Parameters
    ----------
    knots
        Knots of the spline.
    degree
        Degree of the spline.
    ldegree
        Left extrapolation polynomial degree.
    rdegree
        Right extrapolation polynomial degree.
    coef
        The coefficients for linear combining the spline basis.

    """

    def __init__(
        self,
        knots: tuple[float, ...],
        degree: int,
        ldegree: Optional[int] = None,
        rdegree: Optional[int] = None,
        coef: Optional[NDArray] = None,
    ) -> None:
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
        super().__init__(funs, coef=coef)

    def get_design_mat(
        self, x: NDArray, order: int = 0, check_args: bool = True
    ) -> NDArray:
        """Create design matrix from spline basis functions.

        Parameters
        ----------
        x
            Data points.
        order
            Order of differentiation/integration.
        check_args
            If ``True``, it will automatically check and parse the arguments.

        Returns
        -------
        describe
            Design matrix from spline basis functions.

        """
        design_mat = super().get_design_mat(x, order, check_args)
        clear_bspl_cache()
        return design_mat
