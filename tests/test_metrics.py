"""Tests for the holistic evaluation metrics."""

import numpy as np
import pytest

from ezmodel.util.metrics import (
    CALIBRATION,
    METRICS,
    POINT_METRICS,
    calc_metric,
    coverage,
    evaluate,
    get_metric,
    greater_is_better,
    metric_names,
)


def test_backward_compatible_default_metrics():
    # the original four names still resolve and compute on (y, y_hat)
    assert METRICS == ["mse", "mae", "r2", "spear"]
    y = np.random.random(100)
    y_hat = y + np.random.normal(0, 0.01, 100)
    for metric in METRICS:
        assert calc_metric(metric, y, y_hat) is not None


def test_perfect_prediction_extremes():
    y = np.linspace(0, 1, 50)
    # exact prediction: zero error, perfect fit/ranking/selection
    assert calc_metric("rmse", y, y) == pytest.approx(0.0)
    assert calc_metric("mae", y, y) == pytest.approx(0.0)
    assert calc_metric("max_error", y, y) == pytest.approx(0.0)
    assert calc_metric("r2", y, y) == pytest.approx(1.0)
    assert calc_metric("spear", y, y) == pytest.approx(1.0)
    assert calc_metric("kendall", y, y) == pytest.approx(1.0)
    assert calc_metric("regret", y, y) == pytest.approx(0.0)
    assert calc_metric("prec@5", y, y) == pytest.approx(1.0)


def test_direction_metadata():
    # lower-is-better vs higher-is-better is encoded, not guessed
    assert greater_is_better("mae") is False
    assert greater_is_better("rmse") is False
    assert greater_is_better("regret") is False
    assert greater_is_better("nlpd") is False
    assert greater_is_better("r2") is True
    assert greater_is_better("spear") is True
    assert greater_is_better("prec@10") is True


def test_ranking_decoupled_from_magnitude():
    # a constant offset destroys accuracy but preserves ranking perfectly
    y = np.linspace(0, 1, 40)
    y_hat = y + 10.0
    assert calc_metric("mae", y, y_hat) == pytest.approx(10.0)
    assert calc_metric("spear", y, y_hat) == pytest.approx(1.0)
    assert calc_metric("kendall", y, y_hat) == pytest.approx(1.0)


def test_selection_metrics_minimization():
    # smaller y is "better"; a monotone-increasing surrogate ranks the minimum correctly
    y = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    y_hat = y * 2.0 + 1.0
    assert calc_metric("regret", y, y_hat) == pytest.approx(0.0)  # picks the true min
    # a surrogate that inverts the order has positive regret
    assert calc_metric("regret", y, -y_hat) > 0.0


def test_calibration_requires_sigma():
    y = np.random.random(100)
    y_hat = y + np.random.normal(0, 0.1, 100)
    with pytest.raises(ValueError):
        calc_metric("nlpd", y, y_hat)  # probabilistic metric without sigma
    assert calc_metric("nlpd", y, y_hat, sigma=np.full(100, 0.1)) is not None


def test_coverage_well_calibrated():
    # residuals exactly Gaussian with the stated sigma -> ~nominal coverage
    rng = np.random.default_rng(0)
    n, s = 20000, 0.3
    y_hat = np.zeros(n)
    y = y_hat + rng.normal(0, s, n)
    sigma = np.full(n, s)
    assert coverage(y, y_hat, sigma, level=0.9) == pytest.approx(0.9, abs=0.02)
    assert calc_metric("cal_err", y, y_hat, sigma=sigma) == pytest.approx(0.0, abs=0.02)


def test_evaluate_groups_by_family():
    y = np.random.random(60)
    y_hat = y + np.random.normal(0, 0.05, 60)
    sigma = np.full(60, 0.05)

    without = evaluate(y, y_hat)
    assert CALIBRATION not in without  # no sigma -> no calibration family
    assert set(without) == {"accuracy", "fit", "ranking", "selection"}

    with_sigma = evaluate(y, y_hat, sigma=sigma)
    assert CALIBRATION in with_sigma
    assert "nlpd" in with_sigma[CALIBRATION]


def test_registry_filters():
    assert set(POINT_METRICS) == set(metric_names(kind="point"))
    assert metric_names(family=CALIBRATION) == ["nlpd", "crps", "cal_err"]
    assert get_metric("mae").family == "accuracy"
