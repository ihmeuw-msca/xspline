.. image:: https://img.shields.io/pypi/l/xspline
    :target: https://github.com/zhengp0/xspline/blob/main/LICENSE

.. image:: https://img.shields.io/pypi/v/xspline
    :target: https://pypi.org/project/xspline

.. image:: https://img.shields.io/github/actions/workflow/status/zhengp0/xspline/python-build.yml?branch=main
    :target: https://github.com/zhengp0/xspline/actions

.. image:: https://img.shields.io/badge/docs-here-green
    :target: https://zhengp0.github.io/xspline

.. image:: https://codecov.io/gh/zhengp0/xspline/branch/main/graph/badge.svg?token=WUV5OR77N8 
    :target: https://codecov.io/gh/zhengp0/xspline

.. image:: https://app.codacy.com/project/badge/Grade/308cc2771871498fbcdaee3440e56ad0
    :target: https://app.codacy.com/gh/zhengp0/xspline/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade


XSpline
=======

Advanced spline package that provides b-spline bases, their derivatives and integrals.


Installation
------------

XSpline requires python 3.10 or higher. XSpline only depends on ``numpy>=1.25.1``.
It can be installed via

.. code:: bash

    pip install xspline>=0.1.0

For developers, you can clone the repository and install the package in the
development mode.

.. code::

    git clone https://github.com/zhengp0/xspline.git
    cd xspline
    pip install -e ".[test,docs]"


Usage
-----

You can use XSpline as a univariate function or use it to get design matrix.

.. code:: python
    
    import numpy as np
    import matplotlib.pyplot as plt
    from xspline import XSpline

    spline = XSpline(knots=[0, 0.25, 0.5, 0.75, 1], degree=3)
    x = np.arange(0, 1.01, 0.01)


One is to use XSpline as a univariate function. In this case, user must provide
coefficients for the spline bases.

.. code:: python

    np.random.seed(123)
    spline.coef = np.random.randn(len(spline))
    y, design_mat = spline(x), spline.get_design_mat(x)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot(x, y)
    ax[1].plot(x, design_mat)

.. image:: docs/images/readme_usage_0.png

XSpline can be used to obtain derivatives.

.. code:: python

    dy, ddesign_mat = spline(x, order=1), spline.get_design_mat(x, order=1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot(x, dy)
    ax[1].plot(x, ddesign_mat)

.. image:: docs/images/readme_usage_1.png

XSpline can be used to obtain definite integrals.

.. code:: python

    iy, idesign_mat = spline(x, order=-1), spline.get_design_mat(x, order=-1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot(x, iy)
    ax[1].plot(x, idesign_mat)

.. image:: docs/images/readme_usage_2.png

XSpline can extrapolate with different polynomial options

.. code:: python

    np.random.seed(123)
    # constant extrapolation one the left and linear extrapolation on the right
    spline = XSpline(
        knots=[0, 0.25, 0.5, 0.75, 1],
        degree=3,
        ldegree=0,
        rdegree=1,
        coef=np.random.randn(len(spline)),
    )
    x = np.arange(-0.5, 1.51, 0.01)
    y, design_mat = spline(x), spline.get_design_mat(x)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot(x, y)
    ax[1].plot(x, design_mat)
    for i in range(len(ax)):
        ax[i].vlines(
            [0, 1],
            ymin=0,
            ymax=1,
            transform=ax[i].get_xaxis_transform(),
            linestyle="--",
            linewidth=1,
            color="grey",
        )

.. image:: docs/images/readme_usage_3.png
