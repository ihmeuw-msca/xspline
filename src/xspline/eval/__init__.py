from .poly import PolyEval

poly_eval = PolyEval()
"""Polynomial evaluation function.

Parameters
----------
params
    Coefficients of the polynomial. For example `params=(12, -6, 2)` represents
    the polynomial :math:`12x^2 - 6x + 2`.
x
    The points where the polynomial is evaluated.
p
    The order of differentiation or integration. When `p == 0`, it will return
    the values of the polynomial, when `p > 0`, it will return the derivatives,
    and when `p < 0`, it will return the value of definitely integrals.

Returns
-------
NDArray
    Function values, derivatives or integrals depend on the value of `p`.

Example
-------
.. code-block:: python

    import numpy as np
    from xspline.eval import poly_eval

    # polynomial 12*x**2 - 6*x + 2
    params = (12, -6, 2)

    # points where the polynomial will be evaluated
    x = np.linspace(-1.0, 2.0, 101)
    
    # function values of the polynomial
    y = poly_eval(params, x)
    y = poly_eval(params, x, p=0)
    
    # deriatives of the polynomial
    # 24*x - 6
    dy = poly_eval(params, x, p=1)

    # integrals of the polynomial, integrate starting from 1
    # 4*x**3 - 3*x**2 + 2*x - 3
    X = np.vstack([np.ones(x.size, dtype=x.dtype), x])
    iy = poly_eval(params, X, p=-1)

"""
