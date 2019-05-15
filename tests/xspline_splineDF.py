# test file for splineDF function


def xspline_splineDF():
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

    # test splineDF function left linear without extrapolation
    # -------------------------------------------------------------------------
    bs = xspline(knots, degree, l_linear=True)
    x = np.linspace(-1.0, 1.0, 201)

    my_Df = bs.splineDF(x, 0, 1, l_extra=False)

    valid_index = (x >= bs.knots[0]) & (x <= bs.knots[-1])
    tr_Df = np.zeros(x.size)
    tr_Df[valid_index] = -2.0

    ok = ok and np.linalg.norm(my_Df - tr_Df) < tol

    if not ok:
        print('xspline_splineDF: l_linear error.')

    # test splineDF function left linear extrapolation
    # -------------------------------------------------------------------------
    x = np.linspace(-1.0, 0.0, 101)

    my_Df = bs.splineDF(x, 0, 1, l_extra=True)
    tr_Df = -2.0*np.ones(x.size)

    ok = ok and np.linalg.norm(my_Df - tr_Df) < tol

    if not ok:
        print('xspline_splineDF: l_linear and l_extra error.')

    # test splineDF function right linear without extrapolation
    # -------------------------------------------------------------------------
    bs = xspline(knots, degree, r_linear=True)
    x = np.linspace(0.0, 2.0, 201)

    my_Df = bs.splineDF(x, 1, 1, r_extra=False)

    valid_index = (x >= bs.knots[0]) & (x <= bs.knots[-1])
    tr_Df = np.zeros(x.size)
    tr_Df[valid_index] = 2.0

    ok = ok and np.linalg.norm(my_Df - tr_Df) < tol

    if not ok:
        print('xspline_splineDF: r_linear error.')

    # test splineDF function right linear extrapolation
    # -------------------------------------------------------------------------
    x = np.linspace(1.0, 2.0, 101)

    my_Df = bs.splineDF(x, 1, 1, r_extra=True)
    tr_Df = 2.0*np.ones(x.size)

    ok = ok and np.linalg.norm(my_Df - tr_Df) < tol

    if not ok:
        print('xspline_splineDF: r_linear and r_extra error.')

    return ok
