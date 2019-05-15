# test file for intgLinear function


def xspline_intgLinear():
    import numpy as np
    from xspline.__init__ import xspline

    ok = True
    # setup test problem
    # -------------------------------------------------------------------------
    knots = np.linspace(0.0, 1.0, 3)
    degree = 1

    bs = xspline(knots, degree)

    # test intgLinear function
    # -------------------------------------------------------------------------
    a = np.random.rand(20)
    x = a + 1.0

    slope = 2.0
    intercept_at_0 = 1.0

    my_if = bs.intgLinear(a, x, 1, 0.0, intercept_at_0, slope)
    tr_if = 0.5*slope*(x**2 - a**2) + intercept_at_0*(x - a)

    tol = 1e-10
    ok = ok and np.linalg.norm(my_if - tr_if) < tol

    if not ok:
        print('tr_if', tr_if)
        print('my_if', my_if)

    return ok
