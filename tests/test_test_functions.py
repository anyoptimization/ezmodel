"""Tests for the analytic test-function library."""

import numpy as np
import pytest

from ezmodel.util.test_functions import TEST_FUNCTIONS, get_test_function


@pytest.mark.parametrize("name", sorted(TEST_FUNCTIONS))
def test_shape_and_finiteness(name):
    f, xl, xu = get_test_function(name, n_var=3)
    assert xl.shape == (3,) and xu.shape == (3,)
    X = np.random.default_rng(0).uniform(xl, xu, size=(20, 3))
    y = f(X)
    assert y.shape == (20,)
    assert np.all(np.isfinite(y))


@pytest.mark.parametrize(
    "name,argmin",
    [("sphere", 0.0), ("ackley", 0.0), ("rastrigin", 0.0), ("griewank", 0.0), ("rosenbrock", 1.0)],
)
def test_known_optimum(name, argmin):
    f, _, _ = get_test_function(name, n_var=2)
    x_star = np.full((1, 2), argmin)
    assert f(x_star)[0] == pytest.approx(0.0, abs=1e-9)


def test_unknown_name_raises():
    with pytest.raises(ValueError):
        get_test_function("not_a_function")
