# test file for splineF function


def bspline_splineF():
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

    # test splineF function
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

    tr_f = np.vstack([
        np.hstack([lf1, rzeros]),
        np.hstack([lf2, rf1]),
        np.hstack([lzeros, rf2])
        ])

    my_f = np.vstack([bs.splineF(x, i) for i in range(bs.num_spline_bases)])

    tol = 1e-10
    ok = ok and np.linalg.norm(my_f - tr_f) < tol

    if not ok:
        print('tr_f', tr_f)
        print('my_f', my_f)

    return ok
