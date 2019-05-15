# test file for designMat function


def xspline_designMat():
    import sys
    import numpy as np
    sys.path.append('../xspline/')
    from xspline import xspline

    ok = True
    # setup test problem
    # -------------------------------------------------------------------------
    knots = np.linspace(0.0, 1.0, 3)
    degree = 1

    tol = 1e-10

    # test designMat function left linear
    # -------------------------------------------------------------------------
    bs = xspline(knots, degree, l_linear=True)
    x = np.linspace(0.0, 1.0, 201)

    my_mat = bs.designMat(x)
    tr_mat = np.zeros((x.size, 2))
    tr_mat[:, 0] = 2.0 - 2.0*x
    tr_mat[:, 1] = 2.0*x - 1.0

    ok = ok and np.linalg.norm(my_mat - tr_mat) < tol

    if not ok:
        print('xspline_designMat: l_linear error.')

    # test designMat function right linear
    # -------------------------------------------------------------------------
    bs = xspline(knots, degree, r_linear=True)
    x = np.linspace(0.0, 1.0, 201)

    my_mat = bs.designMat(x)
    tr_mat = np.zeros((x.size, 2))
    tr_mat[:, 0] = 1.0 - 2.0*x
    tr_mat[:, 1] = 2.0*x

    ok = ok and np.linalg.norm(my_mat - tr_mat) < tol

    if not ok:
        print('xspline_designMat: r_linear error.')

    return ok
