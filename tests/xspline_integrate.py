# test file for comparing with numerical integrate


def xspline_integrate():
    import numpy as np
    import scipy.integrate as integrate
    from xspline.__init__ import xspline

    ok = True
    # setup test problem
    # -------------------------------------------------------------------------
    knots = np.linspace(0.0, 1.0, 5)
    degree = 1

    bs = xspline(knots, degree, l_linear=True, r_linear=True)

    # test integrate function
    # -------------------------------------------------------------------------
    N = 3
    x = np.linspace(-1.0, 2.0, N)
    a = np.repeat(-1.0, N)

    my_If = bs.splineIF(a, x, 0, 2)

    def func(x):
        return bs.splineIF(-1.0, x, 0, 1)

    tr_If = np.array([integrate.quad(func, a[i], x[i])[0] for i in range(N)])

    tol = 1e-8
    ok = ok and np.linalg.norm(my_If - tr_If) < tol

    if not ok:
        print('tr_If', tr_If)
        print('my_If', my_If)

    return ok
