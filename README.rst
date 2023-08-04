.. image:: https://github.com/zhengp0/xspline/workflows/python-build/badge.svg
    :target: https://github.com/zhengp0/xspline/actions

.. image:: https://badge.fury.io/py/xspline.svg
    :target: https://badge.fury.io/py/xspline

XSpline
=======

Advanced spline package that provides b-spline bases, their derivatives and integrals.


Installation
------------

XSpline can be install via

.. code-block:: bash

    pip install xspline>=0.1.0


Requirements
------------

XSpline requires python 3.10 or higher. XSpline only depends on ``numpy>=1.25.1``.


Usage
-----

There are two main use-cases.

.. code-block:: python
    
    import numpy as np
    import matplotlib.pyplot as plt
    from xspline import XSpline

    spline = XSpline(knots=[0, 0.25, 0.5, 0.75, 1], degree=3)
    x = np.arange(0, 1.01, 0.01)


one is to use XSpline as a univariate function. In this case, user must provide
coefficients for the spline bases.

.. code-block:: python

    np.random.seed(123)
    spline.coef = np.random.randn(len(spline))
    y = spline(x)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(x, y)
    ax.set_title("Usecase 1", loc="left")

.. image:: images/readme_usage_0.png
    :alt: usecase_1

The other is to obtain the design matrix for other modeling uses.

.. code-block:: python

    design_mat = spline.get_design_mat(x)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(x, design_mat)

.. image:: images/readme_usage_1.png
    :alt: usecase_2
