# define the bspline objective class
import numpy as np
from xspline.utils import *


class bspline:
    # -------------------------------------------------------------------------
    def __init__(self, knots, degree):
        '''constructor of the bspline'''
        knots = list(set(knots))
        knots = np.sort(np.array(knots))

        self.knots = knots
        self.degree = degree

        # check the input for the spline
        self.num_knots = knots.size
        self.num_invls = knots.size - 1
        self.num_spline_bases = self.num_invls + self.degree

        assert self.num_invls >= 1
        assert isinstance(self.degree, int) and self.degree >= 0

    # -------------------------------------------------------------------------
    def splineSGen(self, i, p, l_extra=False, r_extra=False):
        '''atom spline support for the ith spline with degree p'''
        assert isinstance(p, int) and p >= 0
        assert isinstance(i, int) and 0 <= i < self.num_invls + p

        cl = self.knots[max(i - p, 0)]
        cr = self.knots[min(i + 1, self.num_invls)]

        if i == 0 and l_extra:
            cl = -np.inf
        if i == p + self.num_invls - 1 and r_extra:
            cr = np.inf

        return [cl, cr]

    def splineS(self, i, l_extra=False, r_extra=False):
        '''spline support for the ith spline'''
        return self.splineSGen(i, self.degree,
                               l_extra=l_extra, r_extra=r_extra)

    # -------------------------------------------------------------------------
    def splineFGen(self, x, i, p, l_extra=False, r_extra=False):
        '''atom spline function for ith spline for degree p'''
        k = self.num_invls + p - 1
        assert isinstance(p, int) and p >= 0
        assert isinstance(i, int) and 0 <= i <= k

        if p == 0:
            invl = self.splineSGen(i, p, l_extra=l_extra, r_extra=r_extra)
            f = indicator(x, invl, r_close=(i == self.num_invls - 1))
            return f

        if i == 0:
            invl = self.splineSGen(0, p, l_extra=l_extra, r_extra=r_extra)
            invl_lin = self.splineSGen(0, p)
            y = indicator(x, invl)
            z = linearR(x, invl_lin)
            return y*(z**p)

        if i == k:
            invl = self.splineSGen(k, p, l_extra=l_extra, r_extra=r_extra)
            invl_lin = self.splineSGen(k, p)
            y = indicator(x, invl, r_close=True)
            z = linearL(x, invl_lin)
            return y*(z**p)

        lf = self.splineFGen(x, i - 1, p - 1,
                             l_extra=l_extra, r_extra=r_extra) *\
            linearL(x, self.splineSGen(i - 1, p - 1))
        rf = self.splineFGen(x, i, p - 1,
                             l_extra=l_extra, r_extra=r_extra) *\
            linearR(x, self.splineSGen(i, p - 1))

        return lf + rf

    def splineF(self, x, i, l_extra=False, r_extra=False):
        '''spline function for ith spline'''
        return self.splineFGen(x, i, self.degree,
                               l_extra=l_extra, r_extra=r_extra)

    # -------------------------------------------------------------------------
    def splineDFGen(self, x, i, p, n, l_extra=False, r_extra=False):
        '''atom spline differentiation function'''
        k = self.num_invls + p - 1
        assert isinstance(p, int) and p >= 0
        assert isinstance(i, int) and 0 <= i <= k
        assert isinstance(n, int) and n >= 0

        if n == 0:
            return self.splineFGen(x, i, p, l_extra=l_extra, r_extra=r_extra)

        if n > p:
            return np.zeros(x.size)

        if p == 0:
            invl = self.splineSGen(i, p, l_extra=l_extra, r_extra=r_extra)
            f = indicator(x, invl, r_close=(i == self.num_invls - 1))
            return f

        if i == 0:
            rdf = 0.0
        else:
            invl = self.splineSGen(i - 1, p - 1)
            d = invl[1] - invl[0]
            f = (x - invl[0])/d
            rdf = f * self.splineDFGen(x, i - 1, p - 1,  n,
                                       l_extra=l_extra, r_extra=r_extra) +\
                n*self.splineDFGen(x, i - 1, p - 1, n - 1,
                                   l_extra=l_extra, r_extra=r_extra) / d

        if i == k:
            ldf = 0.0
        else:
            invl = self.splineSGen(i, p - 1)
            d = invl[0] - invl[1]
            f = (x - invl[1])/d
            ldf = f * self.splineDFGen(x, i, p - 1, n,
                                       l_extra=l_extra, r_extra=r_extra) +\
                n*self.splineDFGen(x, i, p - 1, n - 1,
                                   l_extra=l_extra, r_extra=r_extra) / d

        return ldf + rdf

    def splineDF(self, x, i, n, l_extra=False, r_extra=False):
        '''spline differentiation function'''
        return self.splineDFGen(x, i, self.degree, n,
                                l_extra=l_extra, r_extra=r_extra)

    # -------------------------------------------------------------------------
    def splineIFGen(self, a, x, i, p, n, l_extra=False, r_extra=False):
        '''atom spline integration function'''
        k = self.num_invls + p - 1
        assert isinstance(p, int) and p >= 0
        assert isinstance(i, int) and 0 <= i <= k
        assert isinstance(n, int) and n >= 0

        if n == 0:
            return self.splineFGen(x, i, p, l_extra=l_extra, r_extra=r_extra)

        if p == 0:
            invl = self.splineSGen(i, 0, l_extra=l_extra, r_extra=r_extra)
            return intgIndicator(a, x, n, invl)

        if i == 0:
            rif = 0.0
        else:
            invl = self.splineSGen(i - 1, p - 1)
            d = invl[1] - invl[0]
            f = (x - invl[0]) / d
            rif = f * self.splineIFGen(a, x, i - 1, p - 1,  n,
                                       l_extra=l_extra, r_extra=r_extra) -\
                n * self.splineIFGen(a, x, i - 1, p - 1, n + 1,
                                     l_extra=l_extra, r_extra=r_extra) / d

        if i == k:
            lif = 0.0
        else:
            invl = self.splineSGen(i, p - 1)
            d = invl[0] - invl[1]
            f = (x - invl[1]) / d
            lif = f * self.splineIFGen(a, x, i, p - 1,  n,
                                       l_extra=l_extra, r_extra=r_extra) -\
                n * self.splineIFGen(a, x, i, p - 1, n + 1,
                                     l_extra=l_extra, r_extra=r_extra) / d

        return lif + rif

    def splineIF(self, a, x, i, n, l_extra=False, r_extra=False):
        '''spline integration function'''
        return self.splineIFGen(a, x, i, self.degree, n,
                                l_extra=l_extra, r_extra=r_extra)

    # -------------------------------------------------------------------------
    def designMat(self, x, l_extra=False, r_extra=False):
        '''spline design matrix function'''
        X = np.vstack([
            self.splineF(x, i, l_extra=l_extra, r_extra=r_extra)
            for i in range(self.num_spline_bases)
            ]).T

        return X

    def designDMat(self, x, n, l_extra=False, r_extra=False):
        '''spline derivative design matrix function'''
        DX = np.vstack([
            self.splineDF(x, i, n, l_extra=l_extra, r_extra=r_extra)
            for i in range(self.num_spline_bases)
            ]).T

        return DX

    def designIMat(self, a, x, n, l_extra=False, r_extra=False):
        '''spline integral design matrix function'''
        IX = np.vstack([
            self.splineIF(a, x, i, n, l_extra=l_extra, r_extra=r_extra)
            for i in range(self.num_spline_bases)
            ]).T

        return IX

    # -------------------------------------------------------------------------
    def lastDMatGen(self, i):
        '''highest order of derivarive matrix atom function'''
        if i == 0:
            return np.identity(self.num_invls)

        k = self.num_invls
        D = seqDiffMat(k + i)

        sl = np.repeat(self.knots[:k], [i] + [1]*(k - 1))
        sr = np.repeat(self.knots[1:], [1]*(k - 1) + [i])

        D /= (sr - sl).reshape(i + k - 1, 1)

        return D

    def lastDMat(self):
        '''highest order of derivative matrix'''
        p = self.degree
        D = p * self.lastDMatGen(p)

        for i in range(p - 1, 0, -1):
            D = i * self.lastDMatGen(i).dot(D)

        return D


class xspline:
    # -------------------------------------------------------------------------
    def __init__(self, knots, degree, l_linear=False, r_linear=False):
        '''constructor of the xspline'''
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

        # there are in total 5 pieces functions
        def l_piece(a, x, n):
            return intgLinear(a, x, n, inner_lb, inner_lb_f, inner_lb_Df)

        def m_piece(a, x, n):
            return self.bs.splineIF(a, x, i, n,
                                    l_extra=l_extra, r_extra=r_extra)

        def r_piece(a, x, n):
            return intgLinear(a, x, n, inner_rb, inner_rb_f, inner_rb_Df)

        funcs = [intgZero, l_piece, m_piece, r_piece, intgZero]
        knots = [outer_lb, inner_lb, inner_rb, outer_rb]

        return intgPieces(a, x, n, funcs, knots)

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
