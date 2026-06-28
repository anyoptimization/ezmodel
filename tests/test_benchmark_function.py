"""Tests for the high-level function benchmark (ezmodel.benchmark.benchmark)."""

import numpy as np
from pydacefit.corr import Gaussian, RationalQuadratic
from pydacefit.regr import LinearRegression

from ezmodel.benchmark import BenchmarkResult, benchmark, default_models
from ezmodel.models.knn import KNN
from ezmodel.models.kriging import Kriging


def _f(X):
    return np.sin(3 * X[:, 0]) + np.cos(2 * X[:, 1])


def test_benchmark_runs_and_summarizes():
    res = benchmark(
        _f,
        xl=[-2, -2],
        xu=[2, 2],
        n=40,
        models={"KNN": KNN(), "Kriging[rq]": Kriging(regr=LinearRegression(), corr=RationalQuadratic(alpha=1.0))},
        n_test=200,
        repeats=3,
        seed=1,
    )
    assert isinstance(res, BenchmarkResult)
    # every model collected the point metrics across all repeats
    assert len(res.raw["KNN"]["rmse"]) == 3
    assert "rmse" in res.metrics() and "spear" in res.metrics()
    # ranking covers every model and the report renders
    assert set(res.ranking()) == {"KNN", "Kriging[rq]"}
    assert res.best() in {"KNN", "Kriging[rq]"}
    assert "best model:" in str(res)


def test_reproducible_with_seed():
    kw = dict(
        xl=[-2, -2],
        xu=[2, 2],
        n=30,
        models={"Kriging[rq]": Kriging(regr=LinearRegression(), corr=RationalQuadratic(alpha=1.0))},
        n_test=150,
        repeats=2,
        seed=7,
    )
    a = benchmark(_f, **kw).mean("rmse")["Kriging[rq]"]
    b = benchmark(_f, **kw).mean("rmse")["Kriging[rq]"]
    assert a == b


def test_calibration_only_for_models_exposing_sigma():
    res = benchmark(
        _f,
        xl=[-2, -2],
        xu=[2, 2],
        n=40,
        models={"KNN": KNN(), "Kriging[gauss]": Kriging(regr=LinearRegression(), corr=Gaussian())},
        n_test=200,
        repeats=2,
        seed=1,
    )
    # Kriging exposes sigma -> calibration metrics present; KNN does not
    assert "nlpd" in res.raw["Kriging[gauss]"]
    assert "nlpd" not in res.raw["KNN"]


def test_default_models_available():
    models = default_models()
    assert "KNN" in models
    assert {f"Kriging[rq[{a}]]" for a in (0.1, 0.25, 0.5, 1.0)} <= set(models)
