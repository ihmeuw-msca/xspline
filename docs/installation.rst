============
Installation
============


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