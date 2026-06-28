"""Golden behavior-regression snapshots over ezmodel's core numeric surfaces."""

import numpy as np
import pytest
from pydacefit.corr import Exponential, Gaussian, RationalQuadratic
from pydacefit.regr import LinearRegression

from ezmodel.models.idw import InverseDistanceWeighting
from ezmodel.models.kriging import Kriging
from ezmodel.models.rbf import RBF
from ezmodel.util.metrics import METRICS, calc_metric
from ezmodel.util.transformation.standardization import Standardization
from ezmodel.util.transformation.zero_to_one import ZeroToOneNormalization


def _dataset():
    """Deterministic 2-D training/test split sampled from a fixed analytic function."""
    rng = np.random.default_rng(42)
    X = rng.random((40, 2))
    y = np.sin(3 * X[:, 0]) + np.cos(2 * X[:, 1]) + 0.5 * X[:, 0] * X[:, 1]
    X_test = rng.random((10, 2))
    return X, y, X_test


@pytest.mark.golden
def test_metrics_golden():
    """Each metric over a fixed (y, y_hat) pair must reproduce its baseline value."""
    rng = np.random.default_rng(7)
    y = rng.random(100)
    y_hat = y + rng.normal(0, 0.05, 100)
    return {m: float(calc_metric(m, y, y_hat)) for m in METRICS}


@pytest.mark.golden
@pytest.mark.parametrize(
    "kernel",
    ["linear", "cubic", "gaussian", "mq", "tps"],
    ids=["linear", "cubic", "gaussian", "mq", "tps"],
)
def test_rbf_predict_golden(kernel):
    """RBF predictions on a fixed dataset must not move across refactors."""
    X, y, X_test = _dataset()
    model = RBF(kernel=kernel, tail="linear").fit(X, y)
    return model.predict(X_test).flatten()


# cubic is omitted: it goes non-positive-definite during pydacefit's theta search on
# this data and raises in Cholesky (a pydacefit robustness bug, not an ezmodel change).
@pytest.mark.golden
@pytest.mark.parametrize(
    "corr",
    [Gaussian(), Exponential(), RationalQuadratic(alpha=1.0)],
    ids=["gauss", "exp", "rq"],
)
def test_kriging_predict_golden(corr):
    """Kriging predictions per correlation kernel (incl. the new ``rq``) must not move."""
    X, y, X_test = _dataset()
    # pin the trend explicitly: the baseline was recorded under a linear trend, so it
    # must not silently shift if the Kriging default regr changes (constant vs linear).
    model = Kriging(regr=LinearRegression(), corr=corr).fit(X, y)
    return model.predict(X_test).flatten()


@pytest.mark.golden
def test_idw_predict_golden():
    """IDW predictions on a fixed dataset must not move across refactors."""
    X, y, X_test = _dataset()
    model = InverseDistanceWeighting().fit(X, y)
    return model.predict(X_test).flatten()


@pytest.mark.golden
@pytest.mark.parametrize(
    "norm",
    [ZeroToOneNormalization, Standardization],
    ids=["zero_to_one", "standardization"],
)
def test_normalization_forward_golden(norm):
    """Forward-normalized values on fixed data must reproduce their baseline."""
    rng = np.random.default_rng(123)
    y = rng.random((30, 2))
    return norm().forward(y)
