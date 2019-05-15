# test file for designIMat function


def bspline_designIMat():
    import sys
    import numpy as np
    sys.path.append('../xspline/')
    from bspline import bspline

    ok = True
    # setup test problem
    # -------------------------------------------------------------------------
    knots = np.linspace(0.0, 1.0, 3)
    degree = 1

    bs = bspline(knots, degree)

    # test designIMat function
    # -------------------------------------------------------------------------
    N = 101
    a = np.zeros(N)
    x = np.linspace(0.0, 1.0, N)
    lx = x[x < 0.5]
    rx = x[x >= 0.5]

    lzeros = np.zeros(lx.size)
    rzeros = np.zeros(rx.size)
    lones = np.ones(lx.size)
    rones = np.ones(rx.size)

    tr_IX = np.vstack([
        np.hstack([lx - lx**2, 0.25*rones]),
        np.hstack([lx**2, (rx - 0.5) - (rx - 0.5)**2 + 0.25*rones]),
        np.hstack([lzeros, (rx - 0.5)**2])
        ]).T

    my_IF = bs.designIMat(a, x, 1)

    tol = 1e-10
    ok = ok and np.linalg.norm(my_IF - tr_IX) < tol

    if not ok:
        print('tr_IX', tr_IX)
        print('my_IF', my_IF)

    return ok
