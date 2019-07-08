# test file for outerFlatten function


def utils_outerFlatten():
    import numpy as np
    from xspline.utils import outerFlatten

    ok = True
    # set up the test problem
    # -------------------------------------------------------------------------
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 3.0, 4.0])
    z = np.array([3.0, 4.0, 5.0])

    x_y = np.array([2.0, 3.0,  4.0,
                    4.0, 6.0,  8.0,
                    6.0, 9.0, 12.0])

    x_y_z = np.array([ 6.0,  8.0, 10.0,
                       9.0, 12.0, 15.0,
                      12.0, 16.0, 20.0,
                      12.0, 16.0, 20.0,
                      18.0, 24.0, 30.0,
                      24.0, 32.0, 40.0,
                      18.0, 24.0, 30.0,
                      27.0, 36.0, 45.0,
                      36.0, 48.0, 60.0])

    ok = ok and np.all(x_y == outerFlatten(x, y))
    ok = ok and np.all(x_y_z == outerFlatten(x, y, z))

    return ok
