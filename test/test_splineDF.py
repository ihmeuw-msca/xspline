import numpy as np
from pybs import bspline

ok = True
# create object
# -----------------------------------------------------------------------------
knots = np.linspace(0.0, 1.0, 5)
bs = bspline(knots)
#
N = 501
d = 3
h = 1e-5
x = np.array([0.0, 0.0+h, 1.0-h, 1.0])
y = np.ones(N)

# test degree 3
# -----------------------------------------------------------------------------
dmat = bs.designMat(d, x)

# finite difference calculate derivative
fd_df = np.zeros((2, dmat.shape[1]))
for j in range(dmat.shape[1]):
	fd_df[0][j] = (dmat[ 1,j] - dmat[ 0,j])/h
	fd_df[1][j] = (dmat[-1,j] - dmat[-2,j])/h

# splineDF calculate derivative
my_df = np.zeros((2, dmat.shape[1]))
for j in range(dmat.shape[1]):
	my_df[:,j] = bs.splineDF(d, j+1, 1, np.array([0.0, 1.0]), extrapolate=True)

err = np.linalg.norm(my_df - fd_df)
#
ok = ok and err < 1e-2
#
if ok: print('splineDF: ok.')
else : print('splineDF: error.')