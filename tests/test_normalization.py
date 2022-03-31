import numpy as np

from ezmodel.util.transformation.standardization import Standardization
from ezmodel.util.transformation.zero_to_one import ZeroToOneNormalization


def foward_and_backward(norm, y=None):
    if y is None:
        y = np.random.random((100, 2))
    n = norm.forward(y)
    y_prime = norm.backward(n)
    return y, y_prime


def test_zero_to_one():
    y, y_prime = foward_and_backward(ZeroToOneNormalization())
    np.testing.assert_allclose(y, y_prime)


def test_standardization():
    y, y_prime = foward_and_backward(Standardization())
    np.testing.assert_allclose(y, y_prime)
