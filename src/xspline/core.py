# -*- coding: utf-8 -*-
"""
    core
    ~~~~

    core module contains main functions and classes.
"""
import numpy as np
from . import utils


def bspline_domain(knots, degree, idx, l_extra=False, r_extra=False):
    r"""Compute the support for the spline basis, knots degree and the index of
    the basis.

    Args:
        knots (numpy.ndarray):
        1D array that stores the knots of the splines.

        degree (int):
        A non-negative integer that indicates the degree of the polynomial.

        idx (int):
        A non-negative integer that indicates the index in the spline bases
        list.

        l_extra (bool, optional):
        A optional bool variable indicates that if extrapolate at left end.
        Default to be False.

        r_extra (bool, optional):
        A optional bool variable indicates that if extrapolate at right end.
        Default to be False.

    Returns:
        numpy.ndarray:
        1D array with two elements represents that left and right end of the
        support of the spline basis.
    """
    num_knots = knots.size
    num_intervals = num_knots - 1
    num_splines = num_intervals + degree

    if idx == -1:
        idx = num_splines - 1

    lb = knots[max(idx - degree, 0)]
    ub = knots[min(idx + 1, num_intervals)]

    if idx == 0 and l_extra:
        lb = -np.inf
    if idx == num_splines - 1 and r_extra:
        ub = np.inf

    return np.array([lb, ub])


def bspline_fun(x, knots, degree, idx, l_extra=False, r_extra=False):
    r"""Compute the spline basis.

    Args:
        x (float | numpy.ndarray):
        Scalar or numpy array that store the independent variables.

        knots (numpy.ndarray):
        1D array that stores the knots of the splines.

        degree (int):
        A non-negative integer that indicates the degree of the polynomial.

        idx (int):
        A non-negative integer that indicates the index in the spline bases
        list.

        l_extra (bool, optional):
        A optional bool variable indicates that if extrapolate at left end.
        Default to be False.

        r_extra (bool, optional):
        A optional bool variable indicates that if extrapolate at right end.
        Default to be False.

    Returns:
        float | numpy.ndarray:
        Function values of the corresponding spline bases.
    """
    num_knots = knots.size
    num_intervals = num_knots - 1
    num_splines = num_intervals + degree

    if idx == -1:
        idx = num_splines - 1

    b = bspline_domain(knots, degree, idx, l_extra=l_extra, r_extra=r_extra)

    if degree == 0:
        f = utils.indicator_f(x, b, r_close=(idx == num_splines - 1))
        return f

    if idx == 0:
        b_effect = bspline_domain(knots, degree, idx)
        y = utils.indicator_f(x, b)
        z = utils.linear_rf(x, b_effect)
        return y*(z**degree)

    if idx == num_splines - 1:
        b_effect = bspline_domain(knots, degree, idx)
        y = utils.indicator_f(x, b, r_close=True)
        z = utils.linear_lf(x, b_effect)
        return y*(z**degree)

    lf = bspline_fun(x, knots, degree - 1, idx - 1,
                     l_extra=l_extra, r_extra=r_extra)
    lf *= utils.linear_lf(x, bspline_domain(knots, degree - 1, idx - 1))

    rf = bspline_fun(x, knots, degree - 1, idx,
                     l_extra=l_extra, r_extra=r_extra)
    rf *= utils.linear_rf(x, bspline_domain(knots, degree - 1, idx))

    return lf + rf


def bspline_dfun(x, knots, degree, order, idx, l_extra=False, r_extra=False):
    r"""Compute the derivative of the spline basis.

    Args:
        x (float | numpy.ndarray):
        Scalar or numpy array that store the independent variables.

        knots (numpy.ndarray):
        1D array that stores the knots of the splines.

        degree (int):
        A non-negative integer that indicates the degree of the polynomial.

        order (int):
        A non-negative integer that indicates the order of differentiation.

        idx (int):
        A non-negative integer that indicates the index in the spline bases
        list.

        l_extra (bool, optional):
        A optional bool variable indicates that if extrapolate at left end.
        Default to be False.

        r_extra (bool, optional):
        A optional bool variable indicates that if extrapolate at right end.
        Default to be False.

    Returns:
        float | numpy.ndarray:
        Derivative values of the corresponding spline bases.
    """
    num_knots = knots.size
    num_intervals = num_knots - 1
    num_splines = num_intervals + degree

    if idx == -1:
        idx = num_splines - 1

    if order == 0:
        return bspline_fun(x, knots, degree, idx,
                           l_extra=l_extra, r_extra=r_extra)

    if order > degree:
        if np.isscalar(x):
            return 0.0
        else:
            return np.zeros(len(x))

    if idx == 0:
        rdf = 0.0
    else:
        b = bspline_domain(knots, degree - 1, idx - 1)
        d = b[1] - b[0]
        f = (x - b[0])/d
        rdf = f*bspline_dfun(x, knots, degree - 1, order, idx - 1,
                             l_extra=l_extra, r_extra=r_extra)
        rdf += order*bspline_dfun(x, knots, degree - 1, order - 1, idx - 1,
                                  l_extra=l_extra, r_extra=r_extra)/d

    if idx == num_splines - 1:
        ldf = 0.0
    else:
        b = bspline_domain(knots, degree - 1, idx)
        d = b[0] - b[1]
        f = (x - b[1])/d
        ldf = f*bspline_dfun(x, knots, degree - 1, order, idx,
                             l_extra=l_extra, r_extra=r_extra)
        ldf += order*bspline_dfun(x, knots, degree - 1, order - 1, idx,
                                  l_extra=l_extra, r_extra=r_extra)/d

    return ldf + rdf


def bspline_ifun(a, x, knots, degree, order, idx, l_extra=False, r_extra=False):
    r"""Compute the integral of the spline basis.

    Args:
        a (float | numpy.ndarray):
        Scalar or numpy array that store the starting point of the integration.

        x (float | numpy.ndarray):
        Scalar or numpy array that store the ending point of the integration.

        knots (numpy.ndarray):
        1D array that stores the knots of the splines.

        degree (int):
        A non-negative integer that indicates the degree of the polynomial.

        order (int):
        A non-negative integer that indicates the order of integration.

        idx (int):
        A non-negative integer that indicates the index in the spline bases
        list.

        l_extra (bool, optional):
        A optional bool variable indicates that if extrapolate at left end.
        Default to be False.

        r_extra (bool, optional):
        A optional bool variable indicates that if extrapolate at right end.
        Default to be False.

    Returns:
        float | numpy.ndarray:
        Integral values of the corresponding spline bases.
    """
    num_knots = knots.size
    num_intervals = num_knots - 1
    num_splines = num_intervals + degree

    if idx == -1:
        idx = num_splines - 1

    if order == 0:
        return bspline_fun(x, knots, degree, idx,
                           l_extra=l_extra, r_extra=r_extra)

    if degree == 0:
        b = bspline_domain(knots, degree, idx, l_extra=l_extra, r_extra=r_extra)
        return utils.indicator_if(a, x, order, b)

    if idx == 0:
        rif = 0.0
    else:
        b = bspline_domain(knots, degree - 1, idx - 1)
        d = b[1] - b[0]
        f = (x - b[0]) / d
        rif = f*bspline_ifun(a, x, knots, degree - 1, order, idx - 1,
                             l_extra=l_extra, r_extra=r_extra)
        rif -= order*bspline_ifun(a, x, knots, degree - 1, order + 1, idx - 1,
                                  l_extra=l_extra, r_extra=r_extra)/d

    if idx == num_splines - 1:
        lif = 0.0
    else:
        b = bspline_domain(knots, degree - 1, idx)
        d = b[0] - b[1]
        f = (x - b[1]) / d
        lif = f*bspline_ifun(a, x, knots, degree - 1, order, idx,
                             l_extra=l_extra, r_extra=r_extra)
        lif -= order*bspline_ifun(a, x, knots, degree - 1, order + 1, idx,
                                  l_extra=l_extra, r_extra=r_extra)/d

    return lif + rif


class XSpline:
    """XSpline main class of the package.
    """

    def __init__(self,
                 knots,
                 degree,
                 l_linear=False,
                 r_linear=False,
                 include_first_basis: bool = True):
        r"""Constructor of the XSpline class.

        knots (numpy.ndarray):
        1D numpy array that store the knots, must including that boundary knots.

        degree (int):
        A non-negative integer that indicates the degree of the spline.

        l_linear (bool, optional):
        A bool variable, that if using the linear tail at left end.

        r_linear (bool, optional):
        A bool variable, that if using the linear tail at right end.
        """
        # pre-process the knots vector
        knots = list(set(knots))
        knots = np.sort(np.array(knots))

        self.knots = knots
        self.degree = degree
        self.l_linear = l_linear
        self.r_linear = r_linear
        self.basis_start = int(not include_first_basis)

        # dimensions
        self.num_knots = knots.size
        self.num_intervals = knots.size - 1

        # check inputs
        int_l_linear = int(l_linear)
        int_r_linear = int(r_linear)
        assert self.num_intervals >= 1 + int_l_linear + int_r_linear
        assert isinstance(self.degree, int) and self.degree >= 0

        # create inner knots
        self.inner_knots = self.knots[int_l_linear:
                                      self.num_knots - int_r_linear]
        self.lb = self.knots[0]
        self.ub = self.knots[-1]
        self.inner_lb = self.inner_knots[0]
        self.inner_ub = self.inner_knots[-1]

        self.num_spline_bases = self.inner_knots.size - 1 + self.degree - self.basis_start

    def domain(self, idx, l_extra=False, r_extra=False):
        """Return the support of the XSpline.

        idx (int):
        A non-negative integer that indicates the index in the spline bases
        list.

        l_extra (bool, optional):
        A optional bool variable indicates that if extrapolate at left end.
        Default to be False.

        r_extra (bool, optional):
        A optional bool variable indicates that if extrapolate at right end.
        Default to be False.

        Returns:
            numpy.ndarray:
            1D array with two elements represents that left and right end of the
            support of the spline basis.
        """
        inner_domain = bspline_domain(self.inner_knots,
                                      self.degree,
                                      idx,
                                      l_extra=l_extra,
                                      r_extra=r_extra)
        lb = inner_domain[0]
        ub = inner_domain[1]

        lb = self.lb if inner_domain[0] == self.inner_lb else lb
        ub = self.ub if inner_domain[1] == self.inner_ub else ub

        return np.array([lb, ub])

    def fun(self, x, idx, l_extra=False, r_extra=False):
        r"""Compute the spline basis.

        Args:
            x (float | numpy.ndarray):
            Scalar or numpy array that store the independent variables.

            idx (int):
            A non-negative integer that indicates the index in the spline bases
            list.

            l_extra (bool, optional):
            A optional bool variable indicates that if extrapolate at left end.
            Default to be False.

            r_extra (bool, optional):
            A optional bool variable indicates that if extrapolate at right end.
            Default to be False.

        Returns:
            float | numpy.ndarray:
            Function values of the corresponding spline bases.
        """
        if not self.l_linear and not self.r_linear:
            return bspline_fun(x,
                               self.inner_knots,
                               self.degree,
                               idx,
                               l_extra=l_extra,
                               r_extra=r_extra)

        x_is_scalar = np.isscalar(x)
        if x_is_scalar:
            x = np.array([x])

        f = np.zeros(x.size)
        m_idx = np.array([True] * x.size)

        if self.l_linear:
            l_idx = (x < self.inner_lb) & ((x >= self.lb) | l_extra)
            m_idx &= (x >= self.inner_lb)

            inner_lb_yun = bspline_fun(self.inner_lb,
                                       self.inner_knots,
                                       self.degree,
                                       idx)
            inner_lb_dfun = bspline_dfun(self.inner_lb,
                                         self.inner_knots,
                                         self.degree,
                                         1, idx)

            f[l_idx] = inner_lb_yun + inner_lb_dfun * (x[l_idx] - self.inner_lb)

        if self.r_linear:
            u_idx = (x > self.inner_ub) & ((x <= self.ub) | r_extra)
            m_idx &= (x <= self.inner_ub)

            inner_ub_yun = bspline_fun(self.inner_ub,
                                       self.inner_knots,
                                       self.degree,
                                       idx)
            inner_ub_dfun = bspline_dfun(self.inner_ub,
                                         self.inner_knots,
                                         self.degree,
                                         1, idx)

            f[u_idx] = inner_ub_yun + inner_ub_dfun * (x[u_idx] - self.inner_ub)

        f[m_idx] = bspline_fun(x[m_idx],
                               self.inner_knots,
                               self.degree,
                               idx,
                               l_extra=l_extra,
                               r_extra=r_extra)

        if x_is_scalar:
            return f[0]
        else:
            return f

    def dfun(self, x, order, idx, l_extra=False, r_extra=False):
        r"""Compute the derivative of the spline basis.

        Args:
            x (float | numpy.ndarray):
            Scalar or numpy array that store the independent variables.

            order (int):
            A non-negative integer that indicates the order of differentiation.

            idx (int):
            A non-negative integer that indicates the index in the spline bases
            list.

            l_extra (bool, optional):
            A optional bool variable indicates that if extrapolate at left end.
            Default to be False.

            r_extra (bool, optional):
            A optional bool variable indicates that if extrapolate at right end.
            Default to be False.

        Returns:
            float | numpy.ndarray:
            Derivative values of the corresponding spline bases.
        """
        if order == 0:
            return self.fun(x, idx, l_extra=l_extra, r_extra=r_extra)

        if (not self.l_linear) and (not self.r_linear):
            return bspline_dfun(x,
                                self.knots,
                                self.degree,
                                order,
                                idx,
                                l_extra=l_extra,
                                r_extra=r_extra)

        x_is_scalar = np.isscalar(x)
        if x_is_scalar:
            x = np.array([x])

        dy = np.zeros(x.size)
        m_idx = np.array([True] * x.size)

        if self.l_linear:
            l_idx = (x < self.inner_lb) & ((x >= self.lb) | l_extra)
            m_idx &= (x >= self.inner_lb)

            if order == 1:
                inner_lb_dy = bspline_dfun(self.inner_lb,
                                           self.inner_knots,
                                           self.degree,
                                           order, idx)
                dy[l_idx] = np.repeat(inner_lb_dy, np.sum(l_idx))

        if self.r_linear:
            u_idx = (x > self.inner_ub) & ((x <= self.ub) | r_extra)
            m_idx &= (x <= self.inner_ub)

            if order == 1:
                inner_ub_dy = bspline_dfun(self.inner_ub,
                                           self.inner_knots,
                                           self.degree,
                                           order, idx)
                dy[u_idx] = np.repeat(inner_ub_dy, np.sum(u_idx))

        dy[m_idx] = bspline_dfun(x[m_idx],
                                 self.inner_knots,
                                 self.degree,
                                 order,
                                 idx,
                                 l_extra=l_extra,
                                 r_extra=r_extra)

        if x_is_scalar:
            return dy[0]
        else:
            return dy

    def ifun(self, a, x, order, idx, l_extra=False, r_extra=False):
        r"""Compute the integral of the spline basis.

        Args:
            a (float | numpy.ndarray):
            Scalar or numpy array that store the starting point of the
            integration.

            x (float | numpy.ndarray):
            Scalar or numpy array that store the ending point of the
            integration.

            order (int):
            A non-negative integer that indicates the order of integration.

            idx (int):
            A non-negative integer that indicates the index in the spline bases
            list.

            l_extra (bool, optional):
            A optional bool variable indicates that if extrapolate at left end.
            Default to be False.

            r_extra (bool, optional):
            A optional bool variable indicates that if extrapolate at right end.
            Default to be False.

        Returns:
            float | numpy.ndarray:
            Integral values of the corresponding spline bases.
        """
        if order == 0:
            return self.fun(x, idx, l_extra=l_extra, r_extra=r_extra)

        if (not self.l_linear) and (not self.r_linear):
            return bspline_ifun(a, x,
                                self.knots,
                                self.degree,
                                order,
                                idx,
                                l_extra=l_extra,
                                r_extra=r_extra)
        # verify the inputs
        assert np.all(a <= x)

        # function and derivative values at inner lb and inner rb
        inner_lb_y = bspline_fun(self.inner_lb,
                                 self.inner_knots,
                                 self.degree,
                                 idx)
        inner_ub_y = bspline_fun(self.inner_ub,
                                 self.inner_knots,
                                 self.degree,
                                 idx)
        inner_lb_dy = bspline_dfun(self.inner_lb,
                                   self.inner_knots,
                                   self.degree,
                                   1, idx)
        inner_ub_dy = bspline_dfun(self.inner_ub,
                                   self.inner_knots,
                                   self.degree,
                                   1, idx)

        # there are in total 5 pieces functions
        def l_piece(a, x, order):
            return utils.linear_if(a, x, order,
                                   self.inner_lb, inner_lb_y, inner_lb_dy)

        def m_piece(a, x, order):
            return bspline_ifun(a, x,
                                self.inner_knots,
                                self.degree,
                                order, idx,
                                l_extra=l_extra, r_extra=r_extra)

        def r_piece(a, x, order):
            return utils.linear_if(a, x, order,
                                   self.inner_ub, inner_ub_y, inner_ub_dy)

        def zero_piece(a, x, order):
            if np.isscalar(a) and np.isscalar(x):
                return 0.0
            elif np.isscalar(a):
                return np.zeros(x.size)
            else:
                return np.zeros(a.size)

        funcs = []
        knots = []
        if not l_extra:
            funcs.append(zero_piece)
        if self.l_linear:
            funcs.append(l_piece)
        funcs.append(m_piece)
        if self.r_linear:
            funcs.append(r_piece)
        if not r_extra:
            funcs.append(zero_piece)

        if not l_extra:
            knots.append(self.lb)
            knots.append(self.inner_lb)
        if self.l_linear:
            knots.append(self.inner_lb)
        if self.r_linear:
            knots.append(self.inner_ub)
        if not r_extra:
            knots.append(self.inner_ub)
            knots.append(self.ub)

        knots = np.sort(list(set(knots)))

        return utils.pieces_if(a, x, order, funcs, knots)

    def design_mat(self, x, l_extra=False, r_extra=False):
        r"""Compute the design matrix of spline basis.

        Args:
            x (float | numpy.ndarray):
            Scalar or numpy array that store the independent variables.

            l_extra (bool, optional):
            A optional bool variable indicates that if extrapolate at left end.
            Default to be False.

            r_extra (bool, optional):
            A optional bool variable indicates that if extrapolate at right end.
            Default to be False.

        Returns:
            numpy.ndarray:
            Return design matrix.
        """
        mat = np.vstack([
            self.fun(x, idx, l_extra=l_extra, r_extra=r_extra)
            for idx in range(self.basis_start, self.num_spline_bases)
        ]).T
        return mat

    def design_dmat(self, x, order, l_extra=False, r_extra=False):
        r"""Compute the design matrix of spline basis derivatives.

        Args:
            x (float | numpy.ndarray):
            Scalar or numpy array that store the independent variables.

            order (int):
            A non-negative integer that indicates the order of differentiation.

            l_extra (bool, optional):
            A optional bool variable indicates that if extrapolate at left end.
            Default to be False.

            r_extra (bool, optional):
            A optional bool variable indicates that if extrapolate at right end.
            Default to be False.

        Returns:
            numpy.ndarray:
            Return design matrix.
        """
        dmat = np.vstack([
            self.dfun(x, order, idx, l_extra=l_extra, r_extra=r_extra)
            for idx in range(self.basis_start, self.num_spline_bases)
        ]).T
        return dmat

    def design_imat(self, a, x, order, l_extra=False, r_extra=False):
        r"""Compute the design matrix of the integrals of the spline bases.

        Args:
            a (float | numpy.ndarray):
            Scalar or numpy array that store the starting point of the
            integration.

            x (float | numpy.ndarray):
            Scalar or numpy array that store the ending point of the
            integration.

            order (int):
            A non-negative integer that indicates the order of integration.

            l_extra (bool, optional):
            A optional bool variable indicates that if extrapolate at left end.
            Default to be False.

            r_extra (bool, optional):
            A optional bool variable indicates that if extrapolate at right end.
            Default to be False.

        Returns:
            numpy.ndarray:
            Return design matrix.
        """
        imat = np.vstack([
            self.ifun(a, x, order, idx, l_extra=l_extra, r_extra=r_extra)
            for idx in range(self.basis_start, self.num_spline_bases)
        ]).T
        return imat

    def last_dmat(self):
        """Compute highest order of derivative in domain.

        Returns:
            numpy.ndarray:
            1D array that contains highest order of derivative for intervals.
        """
        # compute the last dmat for the inner domain
        dmat = self.design_dmat(self.inner_knots[:-1], self.degree)

        if self.l_linear:
            dmat = np.vstack((self.design_dmat(np.array([self.inner_lb]), 1),
                              dmat))

        if self.r_linear:
            dmat = np.vstack((dmat,
                              self.design_dmat(np.array([self.inner_ub]), 1)))

        return dmat


class NDXSpline:
    """Multi-dimensional xspline.
    """

    def __init__(self, ndim, knots_list, degree_list,
                 l_linear_list=None,
                 r_linear_list=None,
                 include_first_basis_list=True):
        """Constructor of ndXSpline class

        Args:
            ndim (int):
                Number of dimension.
            knots_list (list{numpy.ndarray}):
                List of knots for every dimension.
            degree_list (list{int}):
                List of degree for every dimension.
            l_linear_list (list{bool} | None, optional):
                List of indicator of if have left linear tail for each
                dimension.
            r_linear_list (list{bool} | None, optional):
                List of indicator of if have right linear tail for each
                dimension.
        """
        self.ndim = ndim
        self.knots_list = knots_list
        self.degree_list = degree_list
        self.l_linear_list = utils.option_to_list(l_linear_list, self.ndim)
        self.r_linear_list = utils.option_to_list(r_linear_list, self.ndim)
        self.include_first_basis_list = utils.option_to_list(include_first_basis_list, self.ndim)

        self.spline_list = [
            XSpline(self.knots_list[i], self.degree_list[i],
                    l_linear=self.l_linear_list[i],
                    r_linear=self.r_linear_list[i],
                    include_first_basis=self.include_first_basis_list[i])
            for i in range(self.ndim)
        ]

        self.num_knots_list = np.array([
            spline.num_knots for spline in self.spline_list])
        self.num_intervals_list = np.array([
            spline.num_intervals for spline in self.spline_list])
        self.num_spline_bases_list = np.array([
            spline.num_spline_bases for spline in self.spline_list])

        self.num_knots = self.num_knots_list.prod()
        self.num_intervals = self.num_intervals_list.prod()
        self.num_spline_bases = self.num_spline_bases_list.prod()

    def design_mat(self, x_list,
                   is_grid=True,
                   l_extra_list=None,
                   r_extra_list=None):
        """Design matrix of the spline basis.

        Args:
            x_list (list{numpy.ndarray}):
                A list of coordinates for each dimension, they should have the
                same dimension or come in matrix form.
            is_grid (bool, optional):
                If `True` treat the coordinates from `x_list` as the grid points
                and compute the mesh grid from it, otherwise, treat each group
                of the coordinates independent.
            l_extra_list (list{bool} | None, optional):
                Indicators of if extrapolate in the left side for each
                dimension.
            r_extra_list (list{bool} | None, optional):
                Indicators of if extrapolate in the right side for each
                dimension.

        Returns:
            numpy.ndarray:
                Design matrix.
        """
        l_extra_list = utils.option_to_list(l_extra_list, self.ndim)
        r_extra_list = utils.option_to_list(r_extra_list, self.ndim)

        assert len(x_list) == self.ndim
        assert len(l_extra_list) == self.ndim
        assert len(r_extra_list) == self.ndim

        mat_list = [spline.design_mat(x_list[i],
                                      l_extra=l_extra_list[i],
                                      r_extra=r_extra_list[i])
                    for i, spline in enumerate(self.spline_list)]

        if is_grid:
            mat = []
            for i in range(self.num_spline_bases):
                index_list = utils.order_to_index(i, self.num_spline_bases_list)
                bases_list = [mat_list[j][:, index_list[j]]
                              for j in range(self.ndim)]
                mat.append(utils.outer_flatten(*bases_list))
        else:
            num_points = x_list[0].size
            assert np.all([x_list[i].size == num_points
                           for i in range(self.ndim)])
            mat = []
            for i in range(self.num_spline_bases):
                index_list = utils.order_to_index(i, self.num_spline_bases_list)
                bases_list = [mat_list[j][:, index_list[j]]
                              for j in range(self.ndim)]
                mat.append(np.prod(bases_list, axis=0))

        return np.ascontiguousarray(np.vstack(mat).T)

    def design_dmat(self, x_list, n_list,
                    is_grid=True,
                    l_extra_list=None,
                    r_extra_list=None):
        """Design matrix of the derivatives of spline basis.

        Args:
            x_list (list{numpy.ndarray}):
                A list of coordinates for each dimension, they should have the
                same dimension or come in matrix form.
            n_list (list{int}):
                A list of integers indicates the order of differentiation for
                each dimension.
            is_grid (bool, optional):
                If `True` treat the coordinates from `x_list` as the grid points
                and compute the mesh grid from it, otherwise, treat each group
                of the coordinates independent.
            l_extra_list (list{bool} | None, optional):
                Indicators of if extrapolate in the left side for each
                dimension.
            r_extra_list (list{bool} | None, optional):
                Indicators of if extrapolate in the right side for each
                dimension.

        Returns:
            numpy.ndarray:
                Differentiation design matrix.
        """
        l_extra_list = utils.option_to_list(l_extra_list, self.ndim)
        r_extra_list = utils.option_to_list(r_extra_list, self.ndim)

        assert len(x_list) == self.ndim
        assert len(n_list) == self.ndim
        assert len(l_extra_list) == self.ndim
        assert len(r_extra_list) == self.ndim

        dmat_list = [spline.design_dmat(x_list[i], n_list[i],
                                        l_extra=l_extra_list[i],
                                        r_extra=r_extra_list[i])
                     for i, spline in enumerate(self.spline_list)]

        if is_grid:
            dmat = []
            for i in range(self.num_spline_bases):
                index_list = utils.order_to_index(i, self.num_spline_bases_list)
                bases_list = [dmat_list[j][:, index_list[j]]
                              for j in range(self.ndim)]
                dmat.append(utils.outer_flatten(*bases_list))
        else:
            num_points = x_list[0].size
            assert np.all([x_list[i].size == num_points
                           for i in range(self.ndim)])
            dmat = []
            for i in range(self.num_spline_bases):
                index_list = utils.order_to_index(i, self.num_spline_bases_list)
                bases_list = [dmat_list[j][:, index_list[j]]
                              for j in range(self.ndim)]
                dmat.append(np.prod(bases_list, axis=0))

        return np.ascontiguousarray(np.vstack(dmat).T)

    def design_imat(self, a_list, x_list, n_list,
                    is_grid=True,
                    l_extra_list=None,
                    r_extra_list=None):
        """Design matrix of the spline basis.

        Args:
            a_list (list{numpy.ndarray}):
                Start of integration of coordinates for each dimension.
            x_list (list{numpy.ndarray}):
                A list of coordinates for each dimension, they should have the
                same dimension or come in matrix form.
            n_list (list{int}):
                A list of integers indicates the order of integration for
                each dimension.
            is_grid (bool, optional):
                If `True` treat the coordinates from `x_list` as the grid points
                and compute the mesh grid from it, otherwise, treat each group
                of the coordinates independent.
            l_extra_list (list{bool} | None, optional):
                Indicators of if extrapolate in the left side for each
                dimension.
            r_extra_list (list{bool} | None, optional):
                Indicators of if extrapolate in the right side for each
                dimension.

        Returns:
            numpy.ndarray:
                Integration design matrix.
        """
        l_extra_list = utils.option_to_list(l_extra_list, self.ndim)
        r_extra_list = utils.option_to_list(r_extra_list, self.ndim)

        assert len(a_list) == self.ndim
        assert len(x_list) == self.ndim
        assert len(n_list) == self.ndim
        assert len(l_extra_list) == self.ndim
        assert len(r_extra_list) == self.ndim

        imat_list = [spline.design_imat(a_list[i], x_list[i], n_list[i],
                                        l_extra=l_extra_list[i],
                                        r_extra=r_extra_list[i])
                     for i, spline in enumerate(self.spline_list)]

        if is_grid:
            imat = []
            for i in range(self.num_spline_bases):
                index_list = utils.order_to_index(i, self.num_spline_bases_list)
                bases_list = [imat_list[j][:, index_list[j]]
                              for j in range(self.ndim)]
                imat.append(utils.outer_flatten(*bases_list))
        else:
            num_points = x_list[0].size
            assert np.all([x_list[i].size == num_points
                           for i in range(self.ndim)])
            imat = []
            for i in range(self.num_spline_bases):
                index_list = utils.order_to_index(i, self.num_spline_bases_list)
                bases_list = [imat_list[j][:, index_list[j]]
                              for j in range(self.ndim)]
                imat.append(np.prod(bases_list, axis=0))

        return np.ascontiguousarray(np.vstack(imat).T)

    def last_dmat(self):
        """Highest order of derivative matrix.

        Returns:
            numpy.ndarray:
                Design matrix contain the highest order of derivative.
        """
        mat_list = [spline.last_dmat() for spline in self.spline_list]

        mat = []
        for i in range(self.num_spline_bases):
            index_list = utils.order_to_index(i, self.num_spline_bases_list)
            bases_list = [mat_list[j][:, index_list[j]]
                          for j in range(self.ndim)]
            mat.append(utils.outer_flatten(*bases_list))

        return np.ascontiguousarray(np.vstack(mat).T)


# TODO:
# 1. bspline function pass in too many default every time
# 2. name of f, df and if
# 3. the way to deal with the scalar vs array.
# 4. keep the naming scheme consistent.
