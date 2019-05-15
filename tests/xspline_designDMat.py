# test file for designDMat function


def xspline_designDMat():
    import numpy as np
    from xspline.__init__ import xspline

    ok = True
    # setup test problem
    # -------------------------------------------------------------------------
    knots = np.linspace(0.0, 1.0, 3)
    degree = 1

    tol = 1e-10

    # test designDMat function left linear
    # -------------------------------------------------------------------------
    bs = xspline(knots, degree, l_linear=True)
    x = np.linspace(0.0, 1.0, 201)

    my_dmat = bs.designDMat(x, 1)
    tr_dmat = np.zeros((x.size, 2))
    tr_dmat[:, 0] = -2.0
    tr_dmat[:, 1] = 2.0

    ok = ok and np.linalg.norm(my_dmat - tr_dmat) < tol

    if not ok:
        print('xspline_designDMat: l_linear error.')

    # test designDMat function right linear
    # -------------------------------------------------------------------------
    bs = xspline(knots, degree, r_linear=True)
    x = np.linspace(0.0, 1.0, 201)

    my_dmat = bs.designDMat(x, 1)
    tr_dmat = np.zeros((x.size, 2))
    tr_dmat[:, 0] = -2.0
    tr_dmat[:, 1] = 2.0

    ok = ok and np.linalg.norm(my_dmat - tr_dmat) < tol

    if not ok:
        print('xspline_designDMat: r_linear error.')

    return ok
