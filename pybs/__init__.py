# define the bspline objective class

# main class
# =============================================================================
class bspline:
	import numpy as np
	# constructor
	# -------------------------------------------------------------------------
	def __init__(self, knots):
		# check the input
		knots = list(set(knots))
		knots = self.np.sort(self.np.array(knots))
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
		if j ==  1  and extrapolate: cl = -self.np.inf
		if j == i+k and extrapolate: cr =  self.np.inf
		#
		return [cl, cr]

	# bpline functions
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
		l = self.splineF(i-1, j-1, x)*self.linFuncR(x, self.splineC(i-1, j-1))
		r = self.splineF(i-1,  j , x)*self.linFuncL(x, self.splineC(i-1,  j ))
		#
		return l + r

	# bpline derivative functions
	# -------------------------------------------------------------------------
	def splineDF(self, i, j, n, x, extrapolate=False):
		# check the input
		self.check(i, j)
		assert isinstance(n, int) and n>=0, 'n: n must be a non-negative num.'
		#
		k = self.k
		t = self.t
		#
		if n == 0: return self.splineF(i, j, x, extrapolate=extrapolate)
		#
		if n > i: return self.np.zeros(x.size)
		#
		# bottom level when degree i is 0
		if i == 0:
			f = self.indiFunc(x, self.splineC(i, j, extrapolate=extrapolate),
				include_r=(j==k))
			return f
		#
		# special cases
		if j == 1:
			r = 0.0
		else:
			c = self.splineC(i-1, j-1)
			d = c[1] - c[0]
			f = (x - c[0])/d
			r = self.splineDF(i-1, j-1,  n , x)*f + n*\
				self.splineDF(i-1, j-1, n-1, x)/d
		#
		if j == k + i:
			l = 0.0
		else:
			c = self.splineC(i-1,  j )
			d = c[0] - c[1]
			f = (x - c[1])/d
			l = self.splineDF(i-1, j,  n , x)*f + n*\
				self.splineDF(i-1, j, n-1, x)/d
		#
		return l + r

	# check function
	# -------------------------------------------------------------------------
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
		X = self.np.zeros((x.size, p+k))
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
		D = p*self.seqDervMat(p)
		for j in range(p-1, p-i, -1):
			D = j*self.seqDervMat(j).dot(D)
		#
		return D

	# utilities
	# -------------------------------------------------------------------------
	def seqDervMat(self, i):
		k = self.k
		t = self.t
		#
		if i == 0: return self.np.eye(k)
		#
		D = self.seqDiffMat(k + i)
		#
		cl = self.np.repeat(t[:k], [i] + [1]*(k-1))
		cr = self.np.repeat(t[1:], [1]*(k-1) + [i])
		#
		D /= (cr - cl).reshape(i + k - 1, 1)
		#
		return D

	def seqDiffMat(self, n):
		assert n>=2 and isinstance(n, int), \
			'n: n must be interger that greater or equal than 2.'
		M = self.np.zeros((n-1, n))
		id_d0 = self.np.diag_indices(n-1)
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

	def linExten(self, t, t0, f0, df0):
		return f0 + df0*(t - t0)

	def indiFunc(self, t, c, include_l=True, include_r=False):
		if include_l: l = t >= c[0]
		else:         l = t >  c[0]
		#
		if include_r: r = t <= c[1]
		else:         r = t <  c[1]
		#
		if self.np.isscalar(t): return float(l&r)
		else:                   return (l&r).astype(self.np.double)


# general function design matrix
# =============================================================================
def designMat(x, knots, degree, l_linear=False, r_linear=False):
	import numpy as np
	#
	knots = np.sort(np.array(list(set(knots))))
	# check the input
	assert knots.size>=2+l_linear+r_linear, \
		'knots: wrong number of knots.'
	assert isinstance(degree, int) and degree>=0, \
		'degree: degree must be non-negative integer.'
	#
	# extrac the inner and outer region
	a = 0
	b = knots.size
	if l_linear: a += 1
	if r_linear: b -= 1
	#
	bs = bspline(knots[a:b])
	#
	lx = x[x < bs.t[0]]
	ix = x[(bs.t[0] <= x) & (x <= bs.t[-1])]
	rx = x[x > bs.t[-1]]
	#
	# design matrix of different parts
	iM = bs.designMat(degree, ix)
	lM = np.zeros((lx.size, iM.shape[1]))
	rM = np.zeros((rx.size, iM.shape[1]))
	for j in range(iM.shape[1]):
		lf = bs.splineF(degree, j+1, bs.t[ 0])
		rf = bs.splineF(degree, j+1, bs.t[-1])
		dlf = bs.splineDF(degree, j+1, 1, bs.t[ 0])
		drf = bs.splineDF(degree, j+1, 1, bs.t[-1])
		# 
		lM[:,j] = bs.linExten(lx, bs.t[ 0], lf, dlf)
		rM[:,j] = bs.linExten(rx, bs.t[-1], rf, drf)
	#
	return np.vstack((lM, iM, rM)), bs
