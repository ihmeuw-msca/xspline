"""
Test SplineBasis Class
"""
import pytest
import numpy as np
from xspline.interval import Interval
from xspline.basis import SplineBasis, SplineSpecs, get_spline_bases

knots = [0.0, 0.5, 1.0]
degree = 1


def fun0(x):
    return np.maximum(1 - 2*x, 0)


def fun1(x):
    return np.minimum(2*x, 2 - 2*x)


def fun2(x):
    return np.maximum(0, 2*x - 1)


funs = [fun0, fun1, fun2]


def dfun0(x):
    y = np.zeros(x.size)
    y[x < 0.5] = -2
    return y


def dfun1(x):
    y = np.zeros(x.size)
    y[x < 0.5] = 2
    y[x >= 0.5] = -2
    return y


def dfun2(x):
    y = np.zeros(x.size)
    y[x >= 0.5] = 2
    return y


dfuns = [dfun0, dfun1, dfun2]


def ifun0(x):
    y = np.full(x.size, 0.25)
    lind = x < 0.5
    y[lind] = x[lind] - x[lind]**2
    return y


def ifun1(x):
    y = np.zeros(x.size)
    lind = x < 0.5
    rind = ~lind
    y[lind] = x[lind]**2
    y[rind] = -x[rind]**2 + 2*x[rind] - 0.5
    return y


def ifun2(x):
    y = np.zeros(x.size)
    rind = x >= 0.5
    y[rind] = x[rind]**2 - x[rind] + 0.25
    return y


ifuns = [ifun0, ifun1, ifun2]


bases = []
for d in range(degree + 1):
    bases.append([
        SplineBasis(SplineSpecs(knots, d, i))
        for i in range(len(knots) + d - 1)
    ])

for d in range(1, degree + 1):
    for i, basis in enumerate(bases[d]):
        indices = set([
            max(0, i - 1),
            min(i, len(bases[d - 1]) - 1)
        ])
        basis.link_bases([bases[d - 1][j] for j in indices])


@pytest.mark.parametrize("data", [np.random.randn(5) for i in range(5)])
def test_basis_fun(data):
    for i, fun in enumerate(funs):
        assert np.allclose(fun(data), bases[-1][i](data, order=0))


@pytest.mark.parametrize("data", [np.random.randn(5) for i in range(5)])
def test_basis_dfun(data):
    for i, fun in enumerate(dfuns):
        assert np.allclose(fun(data), bases[-1][i](data, order=1))


@pytest.mark.parametrize("data", [np.random.randn(5) for i in range(5)])
def test_basis_ifun(data):
    for i, fun in enumerate(ifuns):
        assert np.allclose(fun(data),
                           bases[-1][i]([np.zeros(data.size), data], order=-1))


@pytest.mark.parametrize("knots", [[0.0, 0.0, 0.0],
                                   ["a", "b", "c"],
                                   ["a", 1, 2]])
@pytest.mark.parametrize("degree", [3])
def test_spline_specs_check_knots(knots, degree):
    with pytest.raises(ValueError):
        SplineSpecs(knots, degree)


@pytest.mark.parametrize("knots", [[0.0, 1.0, 2.0]])
@pytest.mark.parametrize("degree", ["a", -1])
def test_spline_specs_check_degree(knots, degree):
    with pytest.raises(AssertionError):
        SplineSpecs(knots, degree)


@pytest.mark.parametrize("knots", [[0, 1, 2],
                                   [0.0, 0.0, 1.0, 2.0],
                                   (0.0, 1.0, 1.0, 2.0)])
@pytest.mark.parametrize("degree", [0, 1, 2])
def test_spline_specs(knots, degree):
    specs = SplineSpecs(knots, degree)
    assert specs.num_bases == degree + 2


@pytest.mark.parametrize("knots", [[0, 1, 2]])
@pytest.mark.parametrize("degree", [0])
@pytest.mark.parametrize("index", [-1, 2, 3, 4])
def test_spline_specs_index_check(knots, degree, index):
    with pytest.raises(AssertionError):
        SplineSpecs(knots, degree, index)


@pytest.mark.parametrize("knots", [[0, 1, 2]])
@pytest.mark.parametrize("degree", [0])
@pytest.mark.parametrize("index", [0, 1])
def test_spline_specs_index(knots, degree, index):
    specs = SplineSpecs(knots, degree, index)
    assert specs.index == index
    assert isinstance(specs.domain, Interval)
    assert isinstance(specs.support, Interval)
    specs.index = None
    assert specs.index is None
    assert specs.domain is None
    assert specs.support is None


@pytest.mark.parametrize("knots", [[0, 1, 2]])
@pytest.mark.parametrize("degree", [0])
@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("with_index", [True, False])
def test_spline_specs_copy(knots, degree, index, with_index):
    specs = SplineSpecs(knots, degree, index)
    specs_copy = specs.copy(with_index=with_index)
    assert np.allclose(specs.knots, specs_copy.knots)
    assert specs.degree == specs_copy.degree
    if with_index:
        assert specs.index == specs_copy.index
    else:
        assert specs_copy.index is None


@pytest.mark.parametrize("specs", [SplineSpecs([0.0, 0.5, 1.0], 3)])
def test_get_spline_bases(specs):
    bases = get_spline_bases(specs)
    assert len(bases) == specs.degree + 1
    assert all([all([basis.is_linked() for basis in bases[d]])
                for d in range(len(bases))])
