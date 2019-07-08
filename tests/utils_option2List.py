# test file for option2List function


def utils_option2List():
    import numpy as np
    from xspline.utils import option2List

    ok = True
    # set up the test problem
    # -------------------------------------------------------------------------
    ok = ok and [False, False, False] == option2List(None, 3)

    ok = ok and [False, False, False] == option2List(False, 3)

    ok = ok and [True, True, True] == option2List(True, 3)

    ok = ok and [True, False, True] == option2List([True, False, True], 3)

    return ok
