# test file for splineS function


def bspline_splineS():
    import numpy as np
    from xspline.__init__ import bspline

    ok = True
    # setup test problem
    # -------------------------------------------------------------------------
    knots = np.linspace(0.0, 1.0, 3)
    degree = 3

    bs = bspline(knots, degree)

    # test splineS function
    # -------------------------------------------------------------------------
    a = len(bs.knots) - 2
    tol = 1e-10

    tr_cl = np.repeat(bs.knots[:-1], [bs.degree + 1] + [1]*a)
    tr_cr = np.repeat(bs.knots[1:], [1]*a + [bs.degree + 1])
    tr_c = np.vstack((tr_cl, tr_cr)).T
    my_c = np.array([bs.splineS(i) for i in range(bs.num_spline_bases)])

    ok = ok and np.linalg.norm(my_c - tr_c) < tol

    if not ok:
        print('tr_c', tr_c)
        print('my_c', my_c)

    return ok
