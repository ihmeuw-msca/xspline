# test file for designIMat function


def lbspline_designIMat():
    import sys
    import numpy as np
    sys.path.append('../bspline/')
    from lbspline import lbspline

    ok = True
    # setup test problem
    # -------------------------------------------------------------------------
    knots = np.linspace(0.0, 1.0, 3)
    degree = 1

    tol = 1e-10

    # test designIMat function left linear
    # -------------------------------------------------------------------------
    bs = lbspline(knots, degree, l_linear=True)
    a = np.zeros(201)
    x = np.linspace(0.0, 1.0, 201)

    my_imat = bs.designIMat(a, x, 1)
    tr_imat = np.zeros((x.size, 2))
    tr_imat[:, 0] = 1.0 - (x - 1.0)**2
    tr_imat[:, 1] = x**2 - x

    ok = ok and np.linalg.norm(my_imat - tr_imat) < tol

    if not ok:
        print('lbspline_designIMat: l_linear error.')

    # test designIMat function right linear
    # -------------------------------------------------------------------------
    bs = lbspline(knots, degree, r_linear=True)
    a = np.zeros(201)
    x = np.linspace(0.0, 1.0, 201)

    my_imat = bs.designIMat(a, x, 1)
    tr_imat = np.zeros((x.size, 2))
    tr_imat[:, 0] = x - x**2
    tr_imat[:, 1] = x**2

    ok = ok and np.linalg.norm(my_imat - tr_imat) < tol

    if not ok:
        print('lbspline_designIMat: r_linear error.')

    return ok
