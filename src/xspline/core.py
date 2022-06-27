from typing import List

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import utils
from ._bspline import bspl_der, bspl_int, bspl_val


class XSpline:
    """XSpline main class of the package.

    Parameters
    ----------
    knots
        1D numpy array that store the knots, must including that boundary knots.

    degree
        A non-negative integer that indicates the degree of the spline.

    l_linear
        A bool variable, that if using the linear tail at left end.

    r_linear
        A bool variable, that if using the linear tail at right end.

    """

    def __init__(self,
                 knots: ArrayLike,
                 degree: int,
                 l_linear: bool = False,
                 r_linear: bool = False,
                 include_first_basis: bool = True):
        # pre-process the knots vector
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

    def fun(self, x: ArrayLike, idx: int) -> NDArray:
        """Compute the spline basis.

        Parameters
        ----------        
        x
            Scalar or numpy array that store the independent variables.
        idx
            A non-negative integer that indicates the index in the spline bases
            list.

        Returns
        -------
        NDArray
            Function values of the corresponding spline bases.

        """
        if not self.l_linear and not self.r_linear:
            return bspl_val(
                self.inner_knots,
                self.degree,
                idx - self.degree,
                x
            )

        f = np.zeros(x.size)
        m_idx = np.array([True] * x.size)

        if idx == -1:
            idx = self.inner_knots.size + self.degree - 2

        if self.l_linear:
            l_idx = (x < self.inner_lb)
            m_idx &= (x >= self.inner_lb)

            inner_lb_fun = bspl_val(
                self.inner_knots,
                self.degree,
                idx - self.degree,
                np.array([self.inner_lb])
            )[0]
            inner_lb_dfun = bspl_der(
                self.inner_knots,
                self.degree,
                idx - self.degree,
                1,
                np.array([self.inner_lb])
            )[0]

            f[l_idx] = inner_lb_fun + inner_lb_dfun * (x[l_idx] - self.inner_lb)

        if self.r_linear:
            u_idx = (x > self.inner_ub)
            m_idx &= (x <= self.inner_ub)

            inner_ub_fun = bspl_val(
                self.inner_knots,
                self.degree,
                idx - self.degree,
                np.array([self.inner_ub])
            )[0]
            inner_ub_dfun = bspl_der(
                self.inner_knots,
                self.degree,
                idx - self.degree,
                1,
                np.array([self.inner_ub])
            )[0]

            f[u_idx] = inner_ub_fun + inner_ub_dfun * (x[u_idx] - self.inner_ub)

        f[m_idx] = bspl_val(
            self.inner_knots,
            self.degree,
            idx - self.degree,
            x[m_idx]
        )

        return f

    def dfun(self, x: ArrayLike, order: int, idx: int) -> NDArray:
        """Compute the derivative of the spline basis.

        Parameters
        ----------
        x
            Scalar or numpy array that store the independent variables.
        order
            A non-negative integer that indicates the order of differentiation.
        idx 
            A non-negative integer that indicates the index in the spline bases
            list.

        Returns
        -------
        NDArray
            Derivative values of the corresponding spline bases.

        """
        if order == 0:
            return self.fun(x, idx)

        if (not self.l_linear) and (not self.r_linear):
            return bspl_der(
                self.inner_knots,
                self.degree,
                idx - self.degree,
                order,
                x
            )

        dy = np.zeros(x.size)
        m_idx = np.array([True] * x.size)

        if idx == -1:
            idx = self.inner_knots.size + self.degree - 2

        if self.l_linear:
            l_idx = (x < self.inner_lb)
            m_idx &= (x >= self.inner_lb)

            if order == 1:
                inner_lb_dy = bspl_der(
                    self.inner_knots,
                    self.degree,
                    idx - self.degree,
                    1,
                    np.array([self.inner_lb])
                )[0]
                dy[l_idx] = np.repeat(inner_lb_dy, np.sum(l_idx))

        if self.r_linear:
            u_idx = (x > self.inner_ub)
            m_idx &= (x <= self.inner_ub)

            if order == 1:
                inner_ub_dy = bspl_der(
                    self.inner_knots,
                    self.degree,
                    idx - self.degree,
                    1,
                    np.array([self.inner_ub])
                )[0]
                dy[u_idx] = np.repeat(inner_ub_dy, np.sum(u_idx))

        dy[m_idx] = bspl_der(
            self.inner_knots,
            self.degree,
            idx - self.degree,
            order,
            x[m_idx]
        )

        return dy

    def ifun(
        self,
        a: float | ArrayLike,
        x: float | ArrayLike,
        order: int,
        idx: int
    ) -> float | NDArray:
        """Compute the integral of the spline basis.

        Parameters
        ----------
        a 
            Scalar or numpy array that store the starting point of the
            integration.
        x 
            Scalar or numpy array that store the ending point of the
            integration.
        order 
            A non-negative integer that indicates the order of integration.
        idx 
            A non-negative integer that indicates the index in the spline bases
            list.

        Returns
        -------
        float | NDArray
            Integral values of the corresponding spline bases.

        """
        if order == 0:
            return self.fun(x, idx)

        if (not self.l_linear) and (not self.r_linear):
            ly = bspl_int(
                self.inner_knots,
                self.degree,
                idx - self.degree,
                -order,
                a
            )
            ry = bspl_int(
                self.inner_knots,
                self.degree,
                idx - self.degree,
                -order,
                a
            )
            return ry - ly

        # verify the inputs
        assert np.all(a <= x)

        inner_lb_y = bspl_val(
            self.inner_knots,
            self.degree,
            idx - self.degree,
            np.array([self.inner_lb])
        )[0]
        inner_ub_y = bspl_val(
            self.inner_knots,
            self.degree,
            idx - self.degree,
            np.array([self.inner_ub])
        )[0]
        inner_lb_dy = bspl_der(
            self.inner_knots,
            self.degree,
            idx - self.degree,
            1,
            np.array([self.inner_lb])
        )[0]
        inner_ub_dy = bspl_der(
            self.inner_knots,
            self.degree,
            idx - self.degree,
            1,
            np.array([self.inner_ub])
        )[0]

        # there are in total 5 pieces functions
        def l_piece(a, x, order):
            return utils.linear_if(a, x, order,
                                   self.inner_lb, inner_lb_y, inner_lb_dy)

        def m_piece(a, x, order):
            ry = bspl_int(
                self.inner_knots,
                self.degree,
                idx - self.degree,
                -order,
                x
            )
            ly = bspl_int(
                self.inner_knots,
                self.degree,
                idx - self.degree,
                -order,
                a
            )
            return ry - ly

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
        if self.l_linear:
            funcs.append(l_piece)
        funcs.append(m_piece)
        if self.r_linear:
            funcs.append(r_piece)

        if self.l_linear:
            knots.append(self.inner_lb)
        if self.r_linear:
            knots.append(self.inner_ub)

        knots = np.sort(list(set(knots)))

        return utils.pieces_if(a, x, order, funcs, knots)

    def design_mat(self, x: float | ArrayLike) -> NDArray:
        """Compute the design matrix of spline basis.

        Parameters
        ----------
        x 
            Scalar or numpy array that store the independent variables.

        Returns
        -------
        NDArray
            Return design matrix.

        """
        mat = np.vstack([
            self.fun(x, idx)
            for idx in range(self.basis_start, self.num_spline_bases + self.basis_start)
        ]).T
        return mat

    def design_dmat(self, x: float | ArrayLike, order: int) -> NDArray:
        """Compute the design matrix of spline basis derivatives.

        Parameters
        ----------
        x
            Scalar or numpy array that store the independent variables.
        order 
            A non-negative integer that indicates the order of differentiation.

        Returns
        -------
        NDArray
            Return design matrix.

        """
        dmat = np.vstack([
            self.dfun(x, order, idx)
            for idx in range(self.basis_start, self.num_spline_bases + self.basis_start)
        ]).T
        return dmat

    def design_imat(
        self,
        a: float | ArrayLike,
        x: float | ArrayLike,
        order: int
    ) -> NDArray:
        """Compute the design matrix of the integrals of the spline bases.

        Parameters
        ----------
        a
            Scalar or numpy array that store the starting point of the
            integration.
        x
            Scalar or numpy array that store the ending point of the
            integration.
        order 
            A non-negative integer that indicates the order of integration.

        Returns
        -------
        NDArray
            Return design matrix.

        """
        imat = np.vstack([
            self.ifun(a, x, order, idx)
            for idx in range(self.basis_start, self.num_spline_bases + self.basis_start)
        ]).T
        return imat

    def last_dmat(self):
        """Compute highest order of derivative in domain.

        Returns:
            NDArray:
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
            knots_list (list{NDArray}):
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

    def design_mat(self, x_list: List[float | ArrayLike], is_grid: bool = True):
        """Design matrix of the spline basis.

        Parameters
        ----------
        x_list
            A list of coordinates for each dimension, they should have the
            same dimension or come in matrix form.
        is_grid
            If `True` treat the coordinates from `x_list` as the grid points
            and compute the mesh grid from it, otherwise, treat each group
            of the coordinates independent.

        Returns
        -------
        NDArray
            Design matrix.

        """
        assert len(x_list) == self.ndim

        mat_list = [spline.design_mat(x_list[i])
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

    def design_dmat(self, x_list, n_list, is_grid=True):
        """Design matrix of the derivatives of spline basis.

        Parameters
        ----------
        x_list
            A list of coordinates for each dimension, they should have the
            same dimension or come in matrix form.
        n_list
            A list of integers indicates the order of differentiation for
            each dimension.
        is_grid
            If `True` treat the coordinates from `x_list` as the grid points
            and compute the mesh grid from it, otherwise, treat each group
            of the coordinates independent.

        Returns
        -------
        NDArray:
            Differentiation design matrix.
        """

        assert len(x_list) == self.ndim
        assert len(n_list) == self.ndim

        dmat_list = [spline.design_dmat(x_list[i], n_list[i])
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

    def design_imat(self, a_list, x_list, n_list, is_grid=True):
        """Design matrix of the spline basis.

        Parameters
        ----------
        a_list
            Start of integration of coordinates for each dimension.
        x_list
            A list of coordinates for each dimension, they should have the
            same dimension or come in matrix form.
        n_list
            A list of integers indicates the order of integration for
            each dimension.
        is_grid
            If `True` treat the coordinates from `x_list` as the grid points
            and compute the mesh grid from it, otherwise, treat each group
            of the coordinates independent.
            dimension.

        Returns
        -------
        NDArray
            Integration design matrix.

        """
        assert len(a_list) == self.ndim
        assert len(x_list) == self.ndim
        assert len(n_list) == self.ndim

        imat_list = [spline.design_imat(a_list[i], x_list[i], n_list[i])
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

        Returns
        -------
        NDArray
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
