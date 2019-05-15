# test file for comparing with numerical integrate


def bspline_integrate():
    import sys
    import numpy as np
    import scipy.integrate as integrate
    sys.path.append('../xspline/')
    from bspline import bspline

    ok = True
    # setup test problem
    # -------------------------------------------------------------------------
    knots = np.linspace(0.0, 1.0, 3)
    degree = 1

    bs = bspline(knots, degree)

    # test integrate function
    # -------------------------------------------------------------------------
    x = np.linspace(-1.0, 2.0, 31)
    a = np.repeat(-1.0, 31)

    my_If = bs.splineIF(a, x, 0, 2)

    func = lambda x: bs.splineIF(-1.0, x, 0, 1)
    tr_If = np.array([integrate.quad(func, a[i], x[i])[0] for i in range(31)])

    tol = 1e-8
    ok = ok and np.linalg.norm(my_If - tr_If) < tol

    if not ok:
        print('tr_If', tr_If)
        print('my_If', my_If)

    return ok
