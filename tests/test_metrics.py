import numpy as np

from ezmodel.util.metrics import METRICS, calc_metric


def test_metrics():
    for metric in METRICS:
        y = np.random.random(100)
        y_hat = y + np.random.normal(0, 0.01, 100)

        v = calc_metric(metric, y, y_hat)

        assert v is not None
