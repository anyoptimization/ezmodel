"""Tests for model selection."""

from pydacefit.corr import Gaussian, RationalQuadratic
from pydacefit.regr import LinearRegression

from ezmodel.core.factory import cartesian
from ezmodel.core.partitioning import merge_and_partition
from ezmodel.core.selection import ModelSelection
from ezmodel.models.kriging import Kriging
from ezmodel.util.sample_from_func import sine_function


def test_selection():
    X, y, _X, _y = sine_function(100, 20)

    models = cartesian(
        Kriging, regr={"lin": LinearRegression()}, corr={"gauss": Gaussian(), "rq": RationalQuadratic(alpha=1.0)}
    )
    X, y, partitions = merge_and_partition((X, y), (_X, _y))

    model = ModelSelection(models, refit=False).do(X, y, partitions=partitions)

    assert model is not None


def test_statistics_table():
    X, y, _X, _y = sine_function(100, 20)

    models = cartesian(
        Kriging, regr={"lin": LinearRegression()}, corr={"gauss": Gaussian(), "rq": RationalQuadratic(alpha=1.0)}
    )
    X, y, partitions = merge_and_partition((X, y), (_X, _y))

    selection = ModelSelection(models, refit=False)
    selection.do(X, y, partitions=partitions)

    stats = selection.statistics()

    # one scored row per model; the score column is named after the ranking metric (mae fallback)
    assert list(stats.columns) == ["label", "mae"]
    assert set(stats["label"]) == set(models)
    assert len(stats) == len(models)
