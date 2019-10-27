# utility functions for the bsplinex class
import numpy as np


def indicator_f(x, b, l_close=True, r_close=False):
    """indicator function"""
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
    """linear function with y as the base point"""
    return fz + dfz*(x - z)


def linear_lf(x, b):
    """linear function start from the left point"""
    return (x - b[0])/(b[1] - b[0])


def linear_rf(x, b):
    """linear function start from the right point"""
    return (x - b[1])/(b[0] - b[1])


def zero_if(a, x, order):
    """integrate constant 0"""
    if np.isscalar(x):
        return 0.0
    else:
        return np.zeros(x.size)


def one_if(a, x, order):
    """integrate constant 1"""
    return (x - a)**order/np.math.factorial(order)


def constant_if(a, x, order, c):
    """integrate constant c n times"""
    return c*one_if(a, x, order)


def linear_if(a, x, order, z, fz, dfz):
    """integrate the linear function"""
    fa = fz + dfz*(a - z)
    dfa = dfz

    return dfa*(x - a)**(order + 1)/np.math.factorial(order + 1) + \
        fa*(x - a)**order/np.math.factorial(order)


def integrate_across_pieces(a, x, order, funcs, knots):
    """integrate Across piecewise functions"""
    if len(funcs) == 1:
        return funcs[0](a, x, order)
    else:
        assert np.all(a < knots[0]) and np.all(x > knots[-1])

    b = np.repeat(knots[0], a.size)
    val = integrate_across_pieces(b, x, order, funcs[1:], knots[1:])

    for j in range(order):
        val += funcs[0](a, b, order - j)*(x - b)**j / np.math.factorial(j)

    return val


def pieces_if(a, x, order, funcs, knots):
    """integrate different pieces of the functions"""
    # verify the input
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


def indicator_if(a, x, order, b):
    """integrate indicator function to the order of n"""
    return pieces_if(a, x, order, [zero_if, one_if, zero_if], b)


def seq_diff_mat(size):
    """sequencial difference matrix"""
    assert isinstance(size, int) and size >= 2

    mat = np.zeros((size - 1, size))
    id_d0 = np.diag_indices(size - 1)
    id_d1 = (id_d0[0], id_d0[1] + 1)

    mat[id_d0] = -1.0
    mat[id_d1] = 1.0

    return mat


def index_list(i, sizes):
    """flat index to index list"""
    assert sizes
    assert i < np.prod(sizes)

    ndim = len(sizes)
    n = np.cumprod(np.insert(sizes[::-1], 0, 1))[-2::-1]

    idxes = []
    for j in range(ndim):
        quotient = i // n[j]
        idxes.append(quotient)
        i -= quotient*n[j]

    return idxes


def option_to_list(opt, size):
    """convert default option to option list"""
    if not opt:
        return [False]*size
    else:
        return [True]*size


def outer_flatten(*arg):
    """outer product of multiple vectors and then flatten the result"""
    ndim = len(arg)
    if ndim == 1:
        return arg[0]

    mat = np.outer(arg[0], arg[1])
    vec = mat.reshape(mat.size,)

    if ndim == 2:
        return vec
    else:
        return outer_flatten(vec, *arg[2:])
