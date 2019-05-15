# test file for splineIF function


def bspline_splineIF():
    import numpy as np
    from xspline.__init__ import bspline

    ok = True
    # setup test problem
    # -------------------------------------------------------------------------
    knots = np.linspace(0.0, 1.0, 3)
    degree = 1

    bs = bspline(knots, degree)

    # test splineIF function
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

    tr_if = np.vstack([
        np.hstack([lx - lx**2, 0.25*rones]),
        np.hstack([lx**2, (rx - 0.5) - (rx - 0.5)**2 + 0.25*rones]),
        np.hstack([lzeros, (rx - 0.5)**2])
        ])

    my_if = np.vstack([bs.splineIF(a, x, i, 1)
                       for i in range(bs.num_spline_bases)])

    tol = 1e-10
    ok = ok and np.linalg.norm(my_if - tr_if) < tol

    if not ok:
        print('tr_if', tr_if)
        print('my_if', my_if)

    return ok
