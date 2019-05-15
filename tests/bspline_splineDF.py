# test file for splineDF function


def bspline_splineDF():
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

    # test splineDF function
    # -------------------------------------------------------------------------
    x = np.linspace(0.0, 1.0, 101)
    lx = x[x < 0.5]
    rx = x[x >= 0.5]

    lzeros = np.zeros(lx.size)
    rzeros = np.zeros(rx.size)
    lones = np.ones(lx.size)
    rones = np.ones(rx.size)

    tr_df = np.vstack([
        np.hstack([-2.0*lones, rzeros]),
        np.hstack([2.0*lones, -2.0*rones]),
        np.hstack([lzeros, 2.0*rones])
        ])

    my_df = np.vstack([bs.splineDF(x, i, 1)
                       for i in range(bs.num_spline_bases)])

    tol = 1e-10
    ok = ok and np.linalg.norm(my_df - tr_df) < tol

    if not ok:
        print('tr_df', tr_df)
        print('my_df', my_df)

    return ok
