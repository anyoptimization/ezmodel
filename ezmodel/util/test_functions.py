"""Standard analytic test functions (varied landscapes) for benchmarking surrogate models."""

import numpy as np


def sphere(X):
    """Smooth convex bowl; the easy baseline. Optimum 0 at the origin."""
    return np.sum(X**2, axis=1)


def rosenbrock(X):
    """Curved, ill-conditioned valley — hard to follow. Optimum 0 at all-ones."""
    return np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1.0 - X[:, :-1]) ** 2, axis=1)


def ackley(X):
    """Many shallow local minima around one deep global basin. Optimum 0 at the origin."""
    d = X.shape[1]
    a = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(X**2, axis=1) / d))
    b = -np.exp(np.sum(np.cos(2 * np.pi * X), axis=1) / d)
    return a + b + 20.0 + np.e


def rastrigin(X):
    """Highly multimodal egg-carton; a regular grid of minima. Optimum 0 at the origin."""
    d = X.shape[1]
    return 10.0 * d + np.sum(X**2 - 10.0 * np.cos(2 * np.pi * X), axis=1)


def griewank(X):
    """Multimodal with a gentle global bowl modulated by cosines. Optimum 0 at the origin."""
    d = X.shape[1]
    s = np.sum(X**2, axis=1) / 4000.0
    p = np.prod(np.cos(X / np.sqrt(np.arange(1, d + 1))), axis=1)
    return s - p + 1.0


def sine(X):
    """Separable sum of sines (pydacefit's usage example function)."""
    return np.sum(np.sin(X * 2 * np.pi), axis=1)


# name -> (function, lower bound, upper bound) over the conventional domain
TEST_FUNCTIONS = {
    "sphere": (sphere, -5.12, 5.12),
    "rosenbrock": (rosenbrock, -2.048, 2.048),
    "ackley": (ackley, -32.768, 32.768),
    "rastrigin": (rastrigin, -5.12, 5.12),
    "griewank": (griewank, -50.0, 50.0),
    "sine": (sine, -1.0, 1.0),
}


def get_test_function(name, n_var=2):
    """Return ``(f, xl, xu)`` for a named test function over its conventional box in ``n_var`` dims.

    Parameters
    ----------
    name : str
        One of :data:`TEST_FUNCTIONS`.
    n_var : int
        Input dimensionality.

    Returns
    -------
    tuple
        ``(f, xl, xu)`` where ``f`` maps ``X`` of shape ``(m, n_var)`` to ``(m,)``, and
        ``xl`` / ``xu`` are length-``n_var`` bound arrays.
    """
    if name not in TEST_FUNCTIONS:
        raise ValueError(f"unknown test function '{name}'. Available: {sorted(TEST_FUNCTIONS)}")
    f, lo, hi = TEST_FUNCTIONS[name]
    return f, np.full(n_var, lo), np.full(n_var, hi)
