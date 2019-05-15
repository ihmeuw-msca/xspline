# test file for designIMat function


def xspline_designIMat():
    import numpy as np
    from xspline.__init__ import xspline

    ok = True
    # setup test problem
    # -------------------------------------------------------------------------
    knots = np.linspace(0.0, 1.0, 3)
    degree = 1

    tol = 1e-10

    # test designIMat function left linear
    # -------------------------------------------------------------------------
    bs = xspline(knots, degree, l_linear=True)
    a = np.zeros(201)
    x = np.linspace(0.0, 1.0, 201)

    my_imat = bs.designIMat(a, x, 1)
    tr_imat = np.zeros((x.size, 2))
    tr_imat[:, 0] = 1.0 - (x - 1.0)**2
    tr_imat[:, 1] = x**2 - x

    ok = ok and np.linalg.norm(my_imat - tr_imat) < tol

    if not ok:
        print('xspline_designIMat: l_linear error.')

    # test designIMat function right linear
    # -------------------------------------------------------------------------
    bs = xspline(knots, degree, r_linear=True)
    a = np.zeros(201)
    x = np.linspace(0.0, 1.0, 201)

    my_imat = bs.designIMat(a, x, 1)
    tr_imat = np.zeros((x.size, 2))
    tr_imat[:, 0] = x - x**2
    tr_imat[:, 1] = x**2

    ok = ok and np.linalg.norm(my_imat - tr_imat) < tol

    if not ok:
        print('xspline_designIMat: r_linear error.')

    return ok
