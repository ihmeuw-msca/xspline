# test file for splineF function


def xspline_splineF():
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

    # test splineF function left linear without extrapolation
    # -------------------------------------------------------------------------
    bs = xspline(knots, degree, l_linear=True)
    x = np.linspace(-1.0, 1.0, 201)

    my_f = bs.splineF(x, 0, l_extra=False)

    valid_index = (x >= bs.knots[0]) & (x <= bs.knots[-1])
    tr_f = np.zeros(x.size)
    tr_f[valid_index] = 2.0*(1.0 - x[valid_index])

    ok = ok and np.linalg.norm(my_f - tr_f) < tol

    if not ok:
        print('xspline_splineF: l_linear error.')

    # test splineF function left linear extrapolation
    # -------------------------------------------------------------------------
    x = np.linspace(-1.0, 0.0, 101)

    my_f = bs.splineF(x, 0, l_extra=True)
    tr_f = 2.0*(1.0 - x)

    ok = ok and np.linalg.norm(my_f - tr_f) < tol

    if not ok:
        print('xspline_splineF: l_linear and l_extra error.')

    # test splineF function right linear without extrapolation
    # -------------------------------------------------------------------------
    bs = xspline(knots, degree, r_linear=True)
    x = np.linspace(0.0, 2.0, 201)

    my_f = bs.splineF(x, 1, r_extra=False)

    valid_index = (x >= bs.knots[0]) & (x <= bs.knots[-1])
    tr_f = np.zeros(x.size)
    tr_f[valid_index] = 2.0*x[valid_index]

    ok = ok and np.linalg.norm(my_f - tr_f) < tol

    if not ok:
        print('xspline_splineF: r_linear error.')

    # test splineF function right linear extrapolation
    # -------------------------------------------------------------------------
    x = np.linspace(1.0, 2.0, 101)

    my_f = bs.splineF(x, 1, r_extra=True)
    tr_f = 2.0*x

    ok = ok and np.linalg.norm(my_f - tr_f) < tol

    if not ok:
        print('xspline_splineF: r_linear and r_extra error.')

    return ok
