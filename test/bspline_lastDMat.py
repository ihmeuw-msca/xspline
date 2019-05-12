# test file for lastDMat function


def bspline_lastDMat():
    import sys
    import numpy as np
    sys.path.append('../bspline/')
    from bspline import bspline

    ok = True
    # setup test problem
    # -------------------------------------------------------------------------
    knots = np.linspace(0.0, 1.0, 3)
    degree = 2

    bs = bspline(knots, degree)

    # test lastDMat function
    # -------------------------------------------------------------------------

    tr_df = np.array([
        [8.0, -12.0, 4.0, 0.0],
        [0.0, 4.0, -12.0, 8.0],
        ])

    my_df = bs.lastDMat()

    tol = 1e-10
    ok = ok and np.linalg.norm(my_df - tr_df) < tol

    if not ok:
        print('tr_df', tr_df)
        print('my_df', my_df)

    return ok
