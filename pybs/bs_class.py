import numpy as np
import copy

class bspline:
	# constructor
	# -------------------------------------------------------------------------
	def __init__(self, knots):
		# check the input
		knots = list(set(knots))
		knots = np.sort(np.array(knots))
		#
		self.k = knots.size - 1
		assert self.k>=1, 'knots: must include starting and ending points.'
		#
		self.t = knots

	# create effective intervals
	# -------------------------------------------------------------------------
	def splineC(self, i, j, extrapolate=False):
		# check the input
		self.check(i, j)
		#
		k = self.k
		t = self.t
		#
		cl = t[max(j-i-1, 0)]
		cr = t[min(  j  , k)]
		#
		if j ==  1  and extrapolate: cl = -np.inf
		if j == i+k and extrapolate: cr =  np.inf
		#
		return [cl, cr]

	# create sbpline functions
	# -------------------------------------------------------------------------
	def splineF(self, i, j, x, extrapolate=False):
		# check the input
		self.check(i, j)
		#
		k = self.k
		t = self.t
		#
		# bottom level when degree i is 0
		if i == 0:
			f = self.indiFunc(x, self.splineC(i, j, extrapolate=extrapolate),
				include_r=(j==k))
			return f
		#
		# special cases
		if j == 1:
			y = self.indiFunc(x, self.splineC(0, 1, extrapolate=extrapolate))
			z = self.linFuncL(x, self.splineC(0, 1))
			return y*(z**i)

		if j == k + i:
			y = self.indiFunc(x, self.splineC(0, k, extrapolate=extrapolate),
				include_r=True)
			z = self.linFuncR(x, self.splineC(0, k))
			return y*(z**i)
		#
		# other levels
		l = self.splineF(i-1, j-1, x)*self.linFuncR(x,
			self.splineC(i-1, j-1, extrapolate=extrapolate))
		r = self.splineF(i-1,  j , x)*self.linFuncL(x,
			self.splineC(i-1,  j , extrapolate=extrapolate))
		#
		return l + r

	def check(self, i, j):
		assert isinstance(i,int) and i>=0,\
			'i: i must be an non-negative integer.'
		assert isinstance(i,int) and 1<=j<=self.k+i,\
			'j: j must be an integer between 1 and k+i.'

	# design matrix
	# -------------------------------------------------------------------------
	def designMat(self, p, x, extrapolate=False):
		assert isinstance(p, int) and p>=0,\
			'p: p must be non-negative integer.'
		k = self.k
		X = np.zeros((x.size, p+k))
		for j in range(p+k):
			X[:,j] = self.splineF(p, j+1, x, extrapolate=extrapolate)
		#
		return X

	# derivative matrix
	# -------------------------------------------------------------------------
	def derivativeMat(self, i, p):
		assert isinstance(p, int) and p>=0,\
			'p: p must be non-negative integer.'
		assert isinstance(i, int) and 0<=i<=p,\
			'i: i must be integer that between 0 and p.'
		#
		D = self.seqDervMat(p)
		for j in range(p-1, i-1, -1):
			D = self.seqDervMat(j).dot(D)
		#
		return D

	# utilities
	# -------------------------------------------------------------------------
	def seqDervMat(self, i):
		k = self.k
		t = self.t
		#
		if i == 0: return np.eye(k)
		#
		D = self.seqDiffMat(k + i)
		#
		cl = np.repeat(t[:k], [i] + [1]*(k-1))
		cr = np.repeat(t[1:], [1]*(k-1) + [i])
		#
		D *= (cr - cl).reshape(i + k - 1, 1)
		#
		return D

	def seqDiffMat(self, n):
		assert n>=2 and isinstance(n, int), \
			'n: n must be interger that greater or equal than 2.'
		M = np.zeros((n-1, n))
		id_d0 = np.diag_indices(n-1)
		id_d1 = (id_d0[0], id_d0[1]+1)
		#
		M[id_d0] = -1.0
		M[id_d1] =  1.0
		#
		return M

	def linFuncL(self, t, c):
		l = c[0]
		u = c[1]
		return (t - u)/(l - u)

	def linFuncR(self, t, c):
		l = c[0]
		u = c[1]
		return (t - l)/(u - l)

	def indiFunc(self, t, c, include_l=True, include_r=False):
		if include_l: l = t >= c[0]
		else:         l = t >  c[0]
		#
		if include_r: r = t <= c[1]
		else:         r = t <  c[1]
		#
		if np.isscalar(t): return float(l&r)
		else:              return (l&r).astype(np.double)
