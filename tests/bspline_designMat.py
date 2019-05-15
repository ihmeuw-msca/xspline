# test file for designMat function


def bspline_designMat():
    import numpy as np
    from xspline.__init__ import bspline

    ok = True
    # setup test problem
    # -------------------------------------------------------------------------
    knots = np.linspace(0.0, 1.0, 3)
    degree = 1

    bs = bspline(knots, degree)

    # test designMat function
    # -------------------------------------------------------------------------
    x = np.linspace(0.0, 1.0, 101)
    lx = x[x < 0.5]
    rx = x[x >= 0.5]

    lzeros = np.zeros(lx.size)
    rzeros = np.zeros(rx.size)
    lf1 = (lx - 0.5)/(0.0 - 0.5)
    lf2 = (lx - 0.0)/(0.5 - 0.0)
    rf1 = (rx - 1.0)/(0.5 - 1.0)
    rf2 = (rx - 0.5)/(1.0 - 0.5)

    tr_X = np.vstack([
        np.hstack([lf1, rzeros]),
        np.hstack([lf2, rf1]),
        np.hstack([lzeros, rf2])
        ]).T

    my_X = bs.designMat(x)

    tol = 1e-10
    ok = ok and np.linalg.norm(my_X - tr_X) < tol

    if not ok:
        print('tr_X', tr_X)
        print('my_X', my_X)

    return ok
