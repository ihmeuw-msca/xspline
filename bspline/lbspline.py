# define the lbspline objective class
import numpy as np
from bspline import bspline


class lbspline:
    # -------------------------------------------------------------------------
    def __init__(self, knots, degree, l_linear=False, r_linear=False):
        '''constructor of the lbspline'''
        knots = list(set(knots))
        knots = np.sort(np.array(knots))

        self.knots = knots
        self.degree = degree
        self.l_linear = l_linear
        self.r_linear = r_linear

        # check the input for the spline
        self.num_knots = knots.size
        self.num_invls = knots.size - 1

        int_l_linear = int(l_linear)
        int_r_linear = int(r_linear)
        assert self.num_invls >= 1 + int_l_linear + int_r_linear
        assert isinstance(self.degree, int) and self.degree >= 0

        # create spline member
        self.bs = bspline(knots[int_l_linear:self.num_knots - int_r_linear],
                          degree)
        self.num_spline_bases = self.bs.num_spline_bases

    # -------------------------------------------------------------------------
    def splineS(self, i, l_extra=False, r_extra=False):
        '''spline support for the ith spline'''
        invl = self.bs.spline(i, l_extra=l_extra, r_extra=r_extra)
        if invl[0] == self.bs.knots[0]:
            invl[0] = self.knots[0]
        if invl[1] == self.bs.knots[-1]:
            invl[1] = self.knots[-1]

        return invl

    # -------------------------------------------------------------------------
    def splineF(self, x, i, l_extra=False, r_extra=False):
        '''spline function for ith spline'''
        if not self.l_linear and not self.r_linear:
            return self.bs.splineF(x, i, l_extra=l_extra, r_extra=r_extra)

        outer_lb = self.knots[0]
        outer_rb = self.knots[-1]
        inner_lb = self.bs.knots[0]
        inner_rb = self.bs.knots[-1]

        x_is_scalar = np.isscalar(x)
        if x_is_scalar:
            x = np.array([x])

        f = np.zeros(x.size)
        m_ind = np.array([True]*x.size)

        if self.l_linear:
            l_ind = (x < inner_lb) & ((x >= outer_lb) | l_extra)
            m_ind &= (x >= inner_lb)

            inner_lb_f = self.bs.splineF(inner_lb, i)
            inner_lb_Df = self.bs.splineDF(inner_lb, i, 1)

            f[l_ind] = inner_lb_f + inner_lb_Df*(x[l_ind] - inner_lb)

        if self.r_linear:
            r_ind = (x > inner_rb) & ((x <= outer_rb) | r_extra)
            m_ind &= (x <= inner_rb)

            inner_rb_f = self.bs.splineF(inner_rb, i)
            inner_rb_Df = self.bs.splineDF(inner_rb, i, 1)

            f[r_ind] = inner_rb_f + inner_rb_Df*(x[r_ind] - inner_rb)

        f[m_ind] = self.bs.splineF(x[m_ind], i,
                                   l_extra=l_extra,
                                   r_extra=r_extra)

        if x_is_scalar:
            return f[0]
        else:
            return f

    # -------------------------------------------------------------------------
    def splineDF(self, x, i, n, l_extra=False, r_extra=False):
        '''spline differentiation function'''
        if n == 0:
            return self.splineF(x, i, l_extra=l_extra, r_extra=r_extra)

        if not self.l_linear and not self.r_linear:
            return self.bs.splineDF(x, i, n, l_extra=l_extra, r_extra=r_extra)

        outer_lb = self.knots[0]
        outer_rb = self.knots[-1]
        inner_lb = self.bs.knots[0]
        inner_rb = self.bs.knots[-1]

        x_is_scalar = np.isscalar(x)
        if x_is_scalar:
            x = np.array([x])

        Df = np.zeros(x.size)
        m_ind = np.array([True]*x.size)

        if self.l_linear:
            l_ind = (x < inner_lb) & ((x >= outer_lb) | l_extra)
            m_ind &= (x >= inner_lb)

            if n == 1:
                inner_lb_Df = self.bs.splineDF(inner_lb, i, n)
                Df[l_ind] = np.repeat(inner_lb_Df, np.sum(l_ind))

        if self.r_linear:
            r_ind = (x > inner_rb) & ((x <= outer_rb) | r_extra)
            m_ind &= (x <= inner_rb)

            if n == 1:
                inner_rb_Df = self.bs.splineDF(inner_rb, i, n)
                Df[r_ind] = np.repeat(inner_rb_Df, np.sum(r_ind))

        Df[m_ind] = self.bs.splineDF(x[m_ind], i, n,
                                     l_extra=l_extra,
                                     r_extra=r_extra)

        if x_is_scalar:
            return Df[0]
        else:
            return Df

    # -------------------------------------------------------------------------
    def splineIF(self, a, x, i, n, l_extra=False, r_extra=False):
        '''spline integration function'''
        if n == 0:
            return self.splineF(x, i, l_extra=l_extra, r_extra=r_extra)

        if not self.l_linear and not self.r_linear:
            return self.bs.splineIF(a, x, i, n,
                                    l_extra=l_extra, r_extra=r_extra)

        # verify the inputs
        assert np.all(a <= x)

        outer_lb = self.knots[0]
        outer_rb = self.knots[-1]
        inner_lb = self.bs.knots[0]
        inner_rb = self.bs.knots[-1]

        x_is_scalar = np.isscalar(x)
        if x_is_scalar:
            a = np.array([a])
            x = np.array([x])

        # function and derivative values at inner lb and inner rb
        inner_lb_f = self.bs.splineF(inner_lb, i)
        inner_rb_f = self.bs.splineF(inner_rb, i)
        inner_lb_Df = self.bs.splineDF(inner_lb, i, 1)
        inner_rb_Df = self.bs.splineDF(inner_rb, i, 1)

        # extrapolation
        if l_extra:
            if self.l_linear:
                outer_lb = -np.inf
            else:
                inner_lb = -np.inf

        if r_extra:
            if self.r_linear:
                outer_rb = np.inf
            else:
                inner_rb = np.inf

        # extract different possible situations for a
        a1_ind = a < outer_lb
        a2_ind = (a >= outer_lb) & (a < inner_lb)
        a3_ind = (a >= inner_lb) & (a < inner_rb)
        a4_ind = (a >= inner_rb) & (a < outer_rb)
        a5_ind = a >= outer_rb

        a_ind = [a1_ind, a2_ind, a3_ind, a4_ind, a5_ind]

        # extract different possible situations for x
        x1_ind = x <= outer_lb
        x2_ind = (x > outer_lb) & (x <= inner_lb)
        x3_ind = (x > inner_lb) & (x <= inner_rb)
        x4_ind = (x > inner_rb) & (x <= outer_rb)
        x5_ind = x > outer_rb

        x_ind = [x1_ind, x2_ind, x3_ind, x4_ind, x5_ind]

        # there are in total 5 pieces functions
        def piece1(a, x, n):
            if np.isscalar(a):
                return 0.0
            else:
                return np.zeros(a.size)

        def piece2(a, x, n):
            return self.intgLinear(a, x, n, inner_lb, inner_lb_f, inner_lb_Df)

        def piece3(a, x, n):
            return self.bs.splineIF(a, x, i, n,
                                    l_extra=l_extra,
                                    r_extra=r_extra)

        def piece4(a, x, n):
            return self.intgLinear(a, x, n, inner_rb, inner_rb_f, inner_rb_Df)

        def piece5(a, x, n):
            if np.isscalar(a):
                return 0.0
            else:
                return np.zeros(a.size)

        funcs = [piece1, piece2, piece3, piece4, piece5]
        knots = [outer_lb, inner_lb, inner_rb, outer_rb]

        If = np.zeros(x.size)

        # in total 15 cases
        for ia in range(5):
            for ix in range(ia, 5):
                case_ind = a_ind[ia] & x_ind[ix]
                if np.any(case_ind):
                    If[case_ind] = self.intgPieces(a[case_ind], x[case_ind], n,
                                                   funcs[ia:ix + 1],
                                                   knots[ia:ix])

        if x_is_scalar:
            return If[0]
        else:
            return If

    # -------------------------------------------------------------------------
    def designMat(self, x, l_extra=False, r_extra=False):
        '''spline design matrix function'''
        X = np.vstack([
            self.splineF(x, i, l_extra=l_extra, r_extra=r_extra)
            for i in range(self.num_spline_bases)
            ]).T

        return np.ascontiguousarray(X)

    def designDMat(self, x, n, l_extra=False, r_extra=False):
        '''spline derivative design matrix function'''
        DX = np.vstack([
            self.splineDF(x, i, n, l_extra=l_extra, r_extra=r_extra)
            for i in range(self.num_spline_bases)
            ]).T

        return np.ascontiguousarray(DX)

    def designIMat(self, a, x, n, l_extra=False, r_extra=False):
        '''spline integral design matrix function'''
        IX = np.vstack([
            self.splineIF(a, x, i, n, l_extra=l_extra, r_extra=r_extra)
            for i in range(self.num_spline_bases)
            ]).T

        return np.ascontiguousarray(IX)

    # -------------------------------------------------------------------------
    def lastDMat(self):
        '''highest order of derivative matrix'''
        inner_lb = self.bs.knots[0]
        inner_rb = self.bs.knots[-1]

        D = self.bs.lastDMat()

        if self.l_linear:
            D = np.vstack((self.bs.designDMat(np.array([inner_lb]), 1), D))

        if self.r_linear:
            D = np.vstack((D, self.bs.designDMat(np.array([inner_rb]), 1)))

        return D

    # -------------------------------------------------------------------------
    @staticmethod
    def intgLinear(a, x, n, y, fy, Dfy):
        '''integrate the linear function'''
        fa = fy + Dfy*(a - y)
        Dfa = Dfy

        return Dfa*(x - a)**(n + 1) / np.math.factorial(n + 1) + \
            fa*(x - a)**n / np.math.factorial(n)

    @staticmethod
    def intgPieces(a, x, n, funcs, knots):
        '''integrate piecewise functions'''
        assert (np.isscalar(a) and np.isscalar(x)) or (a.size == x.size)
        assert len(funcs) == len(knots) + 1
        if len(funcs) == 1:
            return funcs[0](a, x, n)
        else:
            assert np.all(a < knots[0]) and np.all(x > knots[-1])

        if np.isscalar(a):
            b = knots[0]
        else:
            b = np.repeat(knots[0], a.size)

        val = lbspline.intgPieces(b, x, n, funcs[1:], knots[1:])

        for j in range(n):
            val += funcs[0](a, b, n - j)*(x - b)**j / np.math.factorial(j)

        return val
