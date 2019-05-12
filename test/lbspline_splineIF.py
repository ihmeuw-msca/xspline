# test file for splineIF function


def lbspline_splineIF():
    import sys
    import numpy as np
    sys.path.append('../bspline/')
    from lbspline import lbspline

    ok = True
    # setup test problem
    # -------------------------------------------------------------------------
    knots = np.linspace(0.0, 1.0, 3)
    degree = 1

    tol = 1e-10

    # test splineIF function left linear without extrapolation
    # -------------------------------------------------------------------------
    bs = lbspline(knots, degree, l_linear=True)
    a = -1.0*np.ones(201)
    x = np.linspace(-1.0, 1.0, 201)

    my_If = bs.splineIF(a, x, 0, 1, l_extra=False)

    valid_index = (x >= bs.knots[0]) & (x <= bs.knots[-1])
    tr_If = np.zeros(x.size)
    tr_If[valid_index] = 1.0 - (x[valid_index] - 1.0)**2

    ok = ok and np.linalg.norm(my_If - tr_If) < tol

    if not ok:
        print('lbspline_splineIF: l_linear error.')

    # test splineIF function left linear extrapolation
    # -------------------------------------------------------------------------
    a = -1.0*np.ones(101)
    x = np.linspace(-1.0, 0.0, 101)

    my_If = bs.splineIF(a, x, 0, 1, l_extra=True)
    tr_If = 4.0 - (x - 1.0)**2

    ok = ok and np.linalg.norm(my_If - tr_If) < tol

    if not ok:
        print('lbspline_splineIF: l_linear and l_extra error.')

    # test splineIF function right linear without extrapolation
    # -------------------------------------------------------------------------
    bs = lbspline(knots, degree, r_linear=True)
    a = np.zeros(201)
    x = np.linspace(0.0, 2.0, 201)

    my_If = bs.splineIF(a, x, 1, 1, r_extra=False)

    valid_index = (x >= bs.knots[0]) & (x <= bs.knots[-1])
    tr_If = np.ones(x.size)
    tr_If[valid_index] = x[valid_index]**2

    ok = ok and np.linalg.norm(my_If - tr_If) < tol

    if not ok:
        print('lbspline_splineIF: r_linear error.')

    # test splineIF function right linear extrapolation
    # -------------------------------------------------------------------------
    a = np.zeros(201)
    x = np.linspace(0.0, 2.0, 201)

    my_If = bs.splineIF(a, x, 1, 1, r_extra=True)
    tr_If = x**2

    ok = ok and np.linalg.norm(my_If - tr_If) < tol

    if not ok:
        print('lbspline_splineIF: r_linear and r_extra error.')

    return ok
