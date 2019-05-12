# test file for designDMat function


def bspline_designDMat():
    import sys
    import numpy as np
    sys.path.append('../bspline/')
    from bspline import bspline

    ok = True
    # setup test problem
    # -------------------------------------------------------------------------
    knots = np.linspace(0.0, 1.0, 3)
    degree = 1

    bs = bspline(knots, degree)

    # test designDMat function
    # -------------------------------------------------------------------------
    x = np.linspace(0.0, 1.0, 101)
    lx = x[x < 0.5]
    rx = x[x >= 0.5]

    lzeros = np.zeros(lx.size)
    rzeros = np.zeros(rx.size)
    lones = np.ones(lx.size)
    rones = np.ones(rx.size)

    tr_DX = np.vstack([
        np.hstack([-2.0*lones, rzeros]),
        np.hstack([2.0*lones, -2.0*rones]),
        np.hstack([lzeros, 2.0*rones])
        ]).T

    my_DX = bs.designDMat(x, 1)

    tol = 1e-10
    ok = ok and np.linalg.norm(my_DX - tr_DX) < tol

    if not ok:
        print('tr_DX', tr_DX)
        print('my_DX', my_DX)

    return ok
