# utility functions for the bsplinex class
import numpy as np


def indicator_f(x, b, l_close=True, r_close=False):
    r"""Indicator function for provided interval.

    Args:
        x (float | numpy.ndarray):
        Float scalar or numpy array store the variable(s).

        b (numpy.ndarray):
        1D array with 2 elements represent the left and right end of the
        interval.

        l_close (bool | True, optional):
        Bool variable indicate that if include the left end of the interval.

        r_close (bool | False, optional):
        Bool variable indicate that if include the right end of the interval.

    Returns:
        float | numpy.ndarray:
        Return function value(s) at ``x``. The result has the same shape with
        ``x``. 0 in the result indicate corresponding element in ``x`` is not in
        the interval and 1 means the it is in the interval.
    """
    if l_close:
        lb = (x >= b[0])
    else:
        lb = (x > b[0])

    if r_close:
        rb = (x <= b[1])
    else:
        rb = (x < b[1])

    if np.isscalar(x):
        return float(lb & rb)
    else:
        return (lb & rb).astype(np.double)


def linear_f(x, z, fz, dfz):
    r"""Linear function construct by reference point(s).

    Args:
        x (float | numpy.ndarray):
        Float scalar or numpy array store the variable(s).

        z (float | numpy.ndarray):
        Float scalar or numpy array store the reference point(s). When ``x`` is
        ``np.ndarray``, ``z`` has to have the same shape with ``x``.

        fz (float | numpy.ndarray):
        Float scalar or numpy array store the function value(s) at ``z``. Same
        requirements with ``z``.

        dfz (float | numpy.ndarray):
        Float scalar or numpy array store the function derivative(s) at ``z``.
        Same requirements with ``z``.

    Returns:
        float | numpy.ndarray:
        Return function value(s) at ``x`` with linear function constructed at
        base point(s) ``z``. The result has the same shape with ``x`` when
        ``z`` is scalar or same shape with ``z`` when ``x`` is scalar.
    """
    return fz + dfz*(x - z)


def linear_lf(x, b):
    r"""Linear function constructed by linearly interpolate 0 and 1 from left
    end point to right end point.

    Args:
        x (float | numpy.ndarray):
        Float scalar or numpy array that store the variable(s).

        b (numpy.ndarray):
        1D array with 2 elements represent the left and right end of the
        interval.

    Returns:
        float | numpy.ndarray:
        Return function value(s) at ``x``. The result has the same shape with
        ``x``.
    """
    return (x - b[0])/(b[1] - b[0])


def linear_rf(x, b):
    r"""Linear function constructed by linearly interpolate 0 and 1 from right
    end point to left end point.

    Args:
        x (float | numpy.ndarray):
        Float scalar or numpy array that store the variable(s).

        b (numpy.ndarray):
        1D array with 2 elements represent the left and right end of the
        interval.

    Returns:
        float | numpy.ndarray:
        Return function value(s) at ``x``. The result has the same shape with
        ``x``.
    """
    return (x - b[1])/(b[0] - b[1])


def constant_if(a, x, order, c):
    r"""Integration of constant function.

    Args:
        a (float | numpy.ndarray):
        Starting point(s) of the integration. If both ``a`` and ``x`` are
        ``np.ndarray``, they should have the same shape.

        x (float | numpy.ndarray):
        Ending point(s) of the integration. If both ``a`` and ``x`` are
        ``np.ndarray``, they should have the same shape.

        order (int):
        Non-negative integer number indicate the order of integration. In other
        words, how many time(s) we integrate.

        c (float):
        Constant function value.

    Returns:
        float | numpy.ndarray:
        Integration value(s) of the constant function.
    """
    # determine the result size
    a_is_ndarray = isinstance(a, np.ndarray)
    x_is_ndarray = isinstance(x, np.ndarray)

    if a_is_ndarray and x_is_ndarray:
        assert a.size == x.size

    result_is_ndarray = a_is_ndarray or x_is_ndarray
    if a_is_ndarray:
        result_size = a.size
    elif x_is_ndarray:
        result_size = x.size
    else:
        result_size = 1

    # special case when c is 0
    if c == 0.0:
        if result_is_ndarray:
            return np.zeros(result_size)
        else:
            return 0.0

    return c*(x - a)**order/np.math.factorial(order)


def linear_if(a, x, order, z, fz, dfz):
    r"""Integrate linear function constructed by the reference point(s).

    Args:
        a (float | numpy.ndarray):
        Starting point(s) of the integration. If both ``a`` and ``x`` are
        ``np.ndarray``, they should have the same shape.

        x (float | numpy.ndarray):
        Ending point(s) of the integration. If both ``a`` and ``x`` are
        ``np.ndarray``, they should have the same shape.

        order (int):
        Non-negative integer number indicate the order of integration. In other
        words, how many time(s) we integrate.

        z (float | numpy.ndarray):
        Float scalar or numpy array store the reference point(s). When ``x`` is
        ``np.ndarray``, ``z`` has to have the same shape with ``x``.

        fz (float | numpy.ndarray):
        Float scalar or numpy array store the function value(s) at ``z``. Same
        requirements with ``z``.

        dfz (float | numpy.ndarray):
        Float scalar or numpy array store the function derivative(s) at ``z``.
        Same requirements with ``z``.

    Returns:
        float | numpy.ndarray:
        Integration value(s) of the constant function.
    """
    fa = fz + dfz*(a - z)
    dfa = dfz

    return dfa*(x - a)**(order + 1)/np.math.factorial(order + 1) + \
        fa*(x - a)**order/np.math.factorial(order)


def integrate_across_pieces(a, x, order, funcs, knots):
    r"""Integrate Across piecewise functions.

    Args:
        a (float | numpy.ndarray):
        Starting point(s) of the integration. If both ``a`` and ``x`` are
        ``np.ndarray``, they should have the same shape. ``a`` has to be less
        than ``knots[0]``.

        x (float | numpy.ndarray):
        Ending point(s) of the integration. If both ``a`` and ``x`` are
        ``np.ndarray``, they should have the same shape. ``x`` has to be greater
        than ``knots[-1]``.

        order (int):
        Non-negative integer number indicate the order of integration. In other
        words, how many time(s) we integrate.

        funcs (list):
        List of functions defined on pieces of domain.

        knots (numpy.ndarray):
        1D numpy array contains all the breaking points for the piecewise
        functions.

    Returns:
        float | numpy.ndarray:
        Integration value(s) of the constant function.
    """
    assert len(funcs) == len(knots) + 1
    if len(funcs) == 1:
        return funcs[0](a, x, order)
    else:
        assert np.all(a < knots[0]) and np.all(x > knots[-1]), f"{a} should be in [{knots[0]}, {knots[-1]}]."

    if np.isscalar(a):
        b = knots[0]
    else:
        b = np.repeat(knots[0], a.size)

    val = integrate_across_pieces(b, x, order, funcs[1:], knots[1:])

    for j in range(order):
        val += funcs[0](a, b, order - j)*(x - b)**j / np.math.factorial(j)

    return val


def pieces_if(a, x, order, funcs, knots):
    r"""Integrate pieces of the functions.

    Args:
        a (float | numpy.ndarray):
        Starting point(s) of the integration. If both ``a`` and ``x`` are
        ``np.ndarray``, they should have the same shape.

        x (float | numpy.ndarray):
        Ending point(s) of the integration. If both ``a`` and ``x`` are
        ``np.ndarray``, they should have the same shape.

        order (int):
        Non-negative integer number indicate the order of integration. In other
        words, how many time(s) we integrate.

        funcs (list):
        List of functions defined on pieces of domain.

        knots (numpy.ndarray):
        1D numpy array contains all the breaking points for the piecewise
        functions.

    Returns:
        float | numpy.ndarray:
        Integration value(s) of the constant function.
    """
    # verify the input
    assert len(funcs) == len(knots) + 1
    if np.isscalar(a) and not np.isscalar(x):
        a = np.repeat(a, x.size)
    if np.isscalar(x) and not np.isscalar(a):
        x = np.repeat(x, a.size)
    if np.isscalar(a) and np.isscalar(x):
        a = np.array([a])
        x = np.array([x])
        result_is_scalar = True
    else:
        result_is_scalar = False

    assert a.size == x.size
    assert np.all(a <= x)
    assert len(funcs) == len(knots) + 1

    num_knots = len(knots)

    # different cases
    a_ind = [a < knots[0]] +\
            [(a >= knots[i]) & (a < knots[i + 1])
             for i in range(num_knots - 1)] +\
            [a >= knots[-1]]

    x_ind = [x <= knots[0]] +\
            [(x > knots[i]) & (x <= knots[i + 1])
             for i in range(num_knots - 1)] +\
            [x > knots[-1]]

    int_f = np.zeros(a.size)
    for ia in range(len(funcs)):
        for ix in range(ia, len(funcs)):
            case_id = a_ind[ia] & x_ind[ix]
            if np.any(case_id):
                int_f[case_id] = integrate_across_pieces(a[case_id],
                                                         x[case_id],
                                                         order,
                                                         funcs[ia:ix + 1],
                                                         knots[ia:ix])

    if result_is_scalar:
        return int_f[0]
    else:
        return int_f


def indicator_if(a, x, order, b, l_close=True, r_close=False):
    r"""Integrate indicator function.

    Args:
        a (float | numpy.ndarray):
        Starting point(s) of the integration. If both ``a`` and ``x`` are
        ``np.ndarray``, they should have the same shape.

        x (float | numpy.ndarray):
        Ending point(s) of the integration. If both ``a`` and ``x`` are
        ``np.ndarray``, they should have the same shape.

        order (int):
        Non-negative integer number indicate the order of integration. In other
        words, how many time(s) we integrate.

        b (numpy.ndarray):
        1D array with 2 elements represent the left and right end of the
        interval.

        l_close (bool | True, optional):
        Bool variable indicate that if include the left end of the interval.

        r_close (bool | False, optional):
        Bool variable indicate that if include the right end of the interval.

    Returns:
        float | numpy.ndarray:
        Integration value(s) of the constant function.
    """
    if order == 0:
        return indicator_f(x, b, l_close=l_close, r_close=r_close)
    else:
        return pieces_if(a, x, order,
                         [lambda *params: constant_if(*params, 0.0),
                          lambda *params: constant_if(*params, 1.0),
                          lambda *params: constant_if(*params, 0.0)], b)


def seq_diff_mat(size):
    r"""Compute sequencial difference matrix.

    Args:
        size (int):
        Positive integer that indicate the number of columns of the matrix.
        Number of rows will be 1 less than number of columns.

    Returns:
        ndarray:
        A matrix consist of one and minus one.
    """
    assert isinstance(size, int) and size >= 2

    mat = np.zeros((size - 1, size))
    id_d0 = np.diag_indices(size - 1)
    id_d1 = (id_d0[0], id_d0[1] + 1)

    mat[id_d0] = -1.0
    mat[id_d1] = 1.0

    return mat


def order_to_index(order, shape):
    r"""Compute the index of the element in a high dimensional array provided
    the order of the element.

    Args:
        order: int
        Non-negative integer present the order of the element.

        shape: tuple
        Shape tuple of the high dimensional array.

    Returns:
        tuple:
        The index element in the array.
    """
    assert hasattr(shape, '__iter__')
    assert isinstance(order, int)
    assert 0 <= order < np.prod(shape)

    ndim = len(shape)
    n = np.cumprod(np.insert(shape[::-1], 0, 1))[-2::-1]

    index = []
    for j in range(ndim):
        quotient = order // n[j]
        index.append(quotient)
        order -= quotient*n[j]

    return tuple(index)


def option_to_list(opt, size):
    r"""Convert default option to list of options.

    Args:
        opt (bool | None):
        A single option in the form of bool or None.

        size (int):
        Positive integer indicate the size of the option list.

    Returns:
        list:
        Option list consist of bool elements.
    """
    assert isinstance(size, int)
    assert size > 0
    if not opt:
        return [False]*size
    else:
        return [True]*size


def outer_flatten(*args):
    r"""Outer product of multiple vectors and then flatten the result.

    Args:
        args (list | tuple):
        A list or tuple of 1D numpy arrays.

    Return:
        numpy.ndarray:
        1D numpy array that store the flattened outer product.
    """
    result = np.prod(np.ix_(*args))
    return result.reshape(result.size,)
