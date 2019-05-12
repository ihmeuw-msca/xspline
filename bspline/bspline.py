# define the bspline objective class
import numpy as np


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
            f = self.indicator(x, invl, r_close=(i == self.num_invls - 1))
            return f

        if i == 0:
            invl = self.splineSGen(0, p, l_extra=l_extra, r_extra=r_extra)
            invl_lin = self.splineSGen(0, p)
            y = self.indicator(x, invl)
            z = self.linearR(x, invl_lin)
            return y*(z**p)

        if i == k:
            invl = self.splineSGen(k, p, l_extra=l_extra, r_extra=r_extra)
            invl_lin = self.splineSGen(k, p)
            y = self.indicator(x, invl, r_close=True)
            z = self.linearL(x, invl_lin)
            return y*(z**p)

        lf = self.splineFGen(x, i - 1, p - 1,
                             l_extra=l_extra, r_extra=r_extra) *\
            self.linearL(x, self.splineSGen(i - 1, p - 1))
        rf = self.splineFGen(x, i, p - 1,
                             l_extra=l_extra, r_extra=r_extra) *\
            self.linearR(x, self.splineSGen(i, p - 1))

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
            f = self.indicator(x, invl, r_close=(i == self.num_invls - 1))
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
            return self.intgIndicator(a, x, invl, n)

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
        D = self.seqDiffMat(k + i)

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

    # -------------------------------------------------------------------------
    @staticmethod
    def indicator(x, invl, l_close=True, r_close=False):
        '''indicator function'''
        if l_close:
            lb = (x >= invl[0])
        else:
            lb = (x > invl[0])

        if r_close:
            rb = (x <= invl[1])
        else:
            rb = (x < invl[1])

        if np.isscalar(x):
            return float(lb & rb)
        else:
            return (lb & rb).astype(np.double)

    @staticmethod
    def linearL(x, invl):
        '''linear function start from the left point'''
        return (x - invl[0])/(invl[1] - invl[0])

    @staticmethod
    def linearR(x, invl):
        '''linear function start from the right point'''
        return (x - invl[1])/(invl[0] - invl[1])

    @staticmethod
    def intgIndicator(a, x, invl, n):
        '''integrate indicator function to the order of n'''
        s0 = np.maximum(0.0, np.minimum(invl[1] - invl[0], x - invl[0]))
        s1 = np.maximum(0.0, x - invl[1])

        f = s0**n / np.math.factorial(n)
        for i in range(1, n):
            f += s0**(n - i) / np.math.factorial(n - i) *\
                s1**i / np.math.factorial(i)

        return f

    @staticmethod
    def seqDiffMat(n):
        '''sequencial difference matrix'''
        assert isinstance(n, int) and n >= 2

        M = np.zeros((n - 1, n))
        id_d0 = np.diag_indices(n - 1)
        id_d1 = (id_d0[0], id_d0[1] + 1)

        M[id_d0] = -1.0
        M[id_d1] = 1.0

        return M