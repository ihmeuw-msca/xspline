import numpy as np
from pybs import bspline

ok = True
# create object
# -----------------------------------------------------------------------------
knots = np.linspace(0.0, 1.0, 5)
degree = 1
bs = bspline(knots)
#
N = 501
x = np.linspace(0.0, 1.0, N)
y = np.ones(N)

# test degree 0, 1, 2, 3
# -----------------------------------------------------------------------------
for i in range(4):
	Di = bs.designMat(i, x)
	err = np.linalg.norm(np.sum(Di, axis=1) - y)
	ok = ok and err < 1e-10

if ok: print('splineF: ok.')
else : print('splineF: error.')