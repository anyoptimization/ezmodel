"""Tests for the cartesian model-grid factory."""

import pytest
from pydacefit.corr import Gaussian, RationalQuadratic

from ezmodel.core.factory import cartesian
from ezmodel.models.kriging import Kriging
from ezmodel.models.rbf import RBF


def test_list_axes_str_tokens_and_product():
    models = cartesian(RBF, kernel=["cubic", "gaussian"], tail=["linear"])
    assert set(models) == {"RBF[cubic,linear]", "RBF[gaussian,linear]"}
    assert all(isinstance(m, RBF) for m in models.values())


def test_dict_axes_use_authored_tokens():
    models = cartesian(Kriging, corr={"g": Gaussian(), "rq": RationalQuadratic(0.25)})
    assert set(models) == {"Kriging[g]", "Kriging[rq]"}
    assert all(isinstance(m, Kriging) for m in models.values())


def test_no_axes_yields_bare_class_name():
    assert set(cartesian(Kriging)) == {"Kriging"}


def test_duplicate_axis_tokens_raise():
    # two kernels whose str() collides -> would silently overwrite without the guard
    with pytest.raises(ValueError):
        cartesian(Kriging, corr=[Gaussian(), Gaussian()])
