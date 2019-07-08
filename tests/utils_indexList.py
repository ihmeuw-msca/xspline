# test file for indexList function


def utils_indexList():
    import numpy as np
    from xspline.utils import indexList

    ok = True
    # set up the test problem
    # -------------------------------------------------------------------------
    i = 0
    sizes = [2, 3, 4]
    ok = ok and [0, 0, 0] == indexList(i, sizes)
    # print(indexList(i, sizes))

    i = 4
    sizes = [2, 3, 4]
    ok = ok and [0, 1, 0] == indexList(i, sizes)
    # print(indexList(i, sizes))

    i = 6
    sizes = [2, 3, 4]
    ok = ok and [0, 1, 2] == indexList(i, sizes)
    # print(indexList(i, sizes))

    i = 12
    sizes = [2, 3, 4]
    ok = ok and [1, 0, 0] == indexList(i, sizes)
    # print(indexList(i, sizes))

    return ok
