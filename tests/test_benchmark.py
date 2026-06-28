"""Tests for the model benchmark."""

import numpy as np
from pydacefit.corr import Gaussian, RationalQuadratic
from pydacefit.regr import LinearRegression

from ezmodel.core.benchmark import Benchmark
from ezmodel.core.factory import cartesian
from ezmodel.models.kriging import Kriging
from ezmodel.util.partitioning.crossvalidation import CrossvalidationPartitioning
from ezmodel.util.sample_from_func import sine_function


def test_benchmark():
    X, y, _X, _y = sine_function(100, 20)

    models = cartesian(
        Kriging, regr={"lin": LinearRegression()}, corr={"gauss": Gaussian(), "rq": RationalQuadratic(alpha=1.0)}
    )
    benchmark = Benchmark(models)

    partitions = CrossvalidationPartitioning(5).do(X)
    benchmark.do(X, y, partitions)

    vals = benchmark.statistics()

    assert vals is not None


def test_correlation():
    X, y, _X, _y = sine_function(100, 20)

    models = cartesian(
        Kriging, regr={"lin": LinearRegression()}, corr={"gauss": Gaussian(), "rq": RationalQuadratic(alpha=1.0)}
    )
    benchmark = Benchmark(models)
    benchmark.do(X, y, CrossvalidationPartitioning(5).do(X))

    corr = benchmark.correlation()

    # square matrix over the model names, with a unit diagonal (self-correlation)
    assert set(corr.columns) == set(models)
    assert corr.shape == (len(models), len(models))
    assert np.allclose(np.diag(corr.values), 1.0)
