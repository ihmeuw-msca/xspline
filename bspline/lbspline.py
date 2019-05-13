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
        if not l_extra:
            a = np.maximum(a, outer_lb)
            x = np.maximum(x, outer_lb)
        else:
            if self.l_linear:
                outer_lb = -np.inf
            else:
                inner_lb = -np.inf

        if not r_extra:
            a = np.minimum(a, outer_rb)
            x = np.minimum(x, outer_rb)
        else:
            if self.r_linear:
                outer_rb = np.inf
            else:
                inner_rb = np.inf

        # extract different possible situations for a
        a1_ind = a < inner_lb
        a2_ind = (a >= inner_lb) & (a < inner_rb)
        a3_ind = a >= inner_rb

        # extract different possible situations for x
        x1_ind = x < inner_lb
        x2_ind = (x >= inner_lb) & (x < inner_rb)
        x3_ind = x >= inner_rb

        # in total 6 cases
        case1_ind = a1_ind & x1_ind
        case2_ind = a1_ind & x2_ind
        case3_ind = a1_ind & x3_ind
        case4_ind = a2_ind & x2_ind
        case5_ind = a2_ind & x3_ind
        case6_ind = a3_ind & x3_ind

        If = np.zeros(x.size)

        # define 6 function corresponding to 6 cases
        def case1(a_case, x_case, n):
            return self.intgLinear(a_case, x_case, n,
                                   inner_lb,
                                   inner_lb_f,
                                   inner_lb_Df)

        def case2(a_case, x_case, n):
            val = self.bs.splineIF(inner_lb, x_case, i, n,
                                   l_extra=l_extra,
                                   r_extra=r_extra)

            for j in range(n):
                val += case1(a_case, inner_lb, n - j)*(x_case - inner_lb)**j /\
                    np.math.factorial(j)

            return val

        def case3(a_case, x_case, n):
            val = self.intgLinear(inner_rb, x_case, n,
                                  inner_rb,
                                  inner_rb_f,
                                  inner_rb_Df)

            for j in range(n):
                val += case2(a_case, inner_rb, n - j)*(x_case - inner_rb)**j /\
                    np.math.factorial(j)

            return val

        def case4(a_case, x_case, n):
            val = self.bs.splineIF(a_case, x_case, i, n,
                                   l_extra=l_extra,
                                   r_extra=r_extra)

            return val

        def case5(a_case, x_case, n):
            val = self.intgLinear(inner_rb, x_case, n,
                                  inner_rb,
                                  inner_rb_f,
                                  inner_rb_Df)

            for j in range(n):
                val += case4(a_case, inner_rb, n - j)*(x_case - inner_rb)**j /\
                    np.math.factorial(j)

            return val

        def case6(a_case, x_case, n):
            return self.intgLinear(a_case, x_case, n,
                                   inner_rb,
                                   inner_rb_f,
                                   inner_rb_Df)

        # case 1
        if np.any(case1_ind):
            a_case1 = a[case1_ind]
            x_case1 = x[case1_ind]
            If[case1_ind] = case1(a_case1, x_case1, n)

        # case 2
        if np.any(case2_ind):
            a_case2 = a[case2_ind]
            x_case2 = x[case2_ind]
            If[case2_ind] = case2(a_case2, x_case2, n)

        # case 3
        if np.any(case3_ind):
            a_case3 = a[case3_ind]
            x_case3 = x[case3_ind]
            If[case3_ind] = case3(a_case3, x_case3, n)

        # case 4
        if np.any(case4_ind):
            a_case4 = a[case4_ind]
            x_case4 = x[case4_ind]
            If[case4_ind] = case4(a_case4, x_case4, n)

        # case 5
        if np.any(case5_ind):
            a_case5 = a[case5_ind]
            x_case5 = x[case5_ind]
            If[case5_ind] = case5(a_case5, x_case5, n)

        # case 6
        if np.any(case6_ind):
            a_case6 = a[case6_ind]
            x_case6 = x[case6_ind]
            If[case6_ind] = case6(a_case6, x_case6, n)

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
