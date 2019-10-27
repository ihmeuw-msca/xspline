# -*- coding: utf-8 -*-
"""
    test_utils
    ~~~~~~~~~~

    unit tests for xspline.utils
"""
import numpy as np
import pytest
from xspline import utils


@pytest.mark.parametrize(("x", "l_close", "r_close", "result"),
                         [(0.5, False, False, 1.0),
                          (2.0, False, False, 0.0),
                          (1.0, False, False, 0.0),
                          (1.0, False, True, 1.0),
                          (0.0, False, False, 0.0),
                          (0.0, True, False, 1.0),
                          (np.repeat(0.5, 3), False, False, np.ones(3)),
                          (np.repeat(2.0, 3), False, False, np.zeros(3)),
                          (np.repeat(1.0, 3), False, False, np.zeros(3)),
                          (np.repeat(1.0, 3), False, True, np.ones(3)),
                          (np.repeat(0.0, 3), False, False, np.zeros(3)),
                          (np.repeat(0.0, 3), True, False, np.ones(3))])
def test_utils_indicator_f(x, l_close, r_close, result):
    b = np.array([0.0, 1.0])
    my_result = utils.indicator_f(x, b, l_close=l_close, r_close=r_close)
    assert np.linalg.norm(my_result - result) < 1e-10
