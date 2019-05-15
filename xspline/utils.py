# utility functions for the bsplinex class
import numpy as np


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


def linear(x, y, fy, Dfy):
    '''linear function with y as the base point'''
    return fy + Dfy*(x - y)


def linearL(x, invl):
    '''linear function start from the left point'''
    return (x - invl[0])/(invl[1] - invl[0])


def linearR(x, invl):
    '''linear function start from the right point'''
    return (x - invl[1])/(invl[0] - invl[1])


def intgZero(a, x, n):
    '''integrate constant 0'''
    if np.isscalar(a):
        return 0.0
    else:
        return np.zeros(a.size)


def intgOne(a, x, n):
    '''integrate constant 1'''
    return (x - a)**n/np.math.factorial(n)


def intgConstant(a, x, n, c):
    '''integrate constant c n times'''
    return c*(x - a)**n/np.math.factorial(n)


def intgLinear(a, x, n, y, fy, Dfy):
    '''integrate the linear function'''
    fa = fy + Dfy*(a - y)
    Dfa = Dfy

    return Dfa*(x - a)**(n + 1) / np.math.factorial(n + 1) + \
        fa*(x - a)**n / np.math.factorial(n)


def intgAcrossPieces(a, x, n, funcs, knots):
    '''integrate Across piecewise functions'''
    if len(funcs) == 1:
        return funcs[0](a, x, n)
    else:
        assert np.all(a < knots[0]) and np.all(x > knots[-1])

    b = np.repeat(knots[0], a.size)
    val = intgAcrossPieces(b, x, n, funcs[1:], knots[1:])

    for j in range(n):
        val += funcs[0](a, b, n - j)*(x - b)**j / np.math.factorial(j)

    return val


def intgPieces(a, x, n, funcs, knots):
    '''integrate different pieces of the functions'''
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

    num_funcs = len(funcs)
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

    If = np.zeros(a.size)
    for ia in range(len(funcs)):
        for ix in range(ia, len(funcs)):
            case_id = a_ind[ia] & x_ind[ix]
            if np.any(case_id):
                If[case_id] = intgAcrossPieces(a[case_id], x[case_id], n,
                                               funcs[ia:ix + 1],
                                               knots[ia:ix])

    if result_is_scalar:
        return If[0]
    else:
        return If


def intgIndicator(a, x, n, invl):
    '''integrate indicator function to the order of n'''
    return intgPieces(a, x, n, [intgZero, intgOne, intgZero], invl)


def seqDiffMat(n):
    '''sequencial difference matrix'''
    assert isinstance(n, int) and n >= 2

    M = np.zeros((n - 1, n))
    id_d0 = np.diag_indices(n - 1)
    id_d1 = (id_d0[0], id_d0[1] + 1)

    M[id_d0] = -1.0
    M[id_d1] = 1.0

    return M
