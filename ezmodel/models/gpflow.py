"""GPflow-backed Gaussian process surrogate model (optional dependency)."""

import numpy as np

from ezmodel.core.model import Model
from ezmodel.core.prediction import Prediction

try:
    import gpflow
except:  # noqa: E722  (optional dependency import guard)
    raise Exception("Model not found. Please execute: 'pip install gpflow'")


class GPFlow(Model):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _fit(self, X, y, **kwargs):
        kernel = gpflow.kernels.SquaredExponential()
        m = gpflow.models.GPR(
            data=(X, y), kernel=kernel, mean_function=gpflow.mean_functions.Constant(), noise_variance=1e-03
        )

        opt = gpflow.optimizers.Scipy()
        opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
        self.model = m

    def _predict(self, X, sigma=False, grad=False):
        mean, var = self.model.predict_f(X)
        std = np.sqrt(np.clip(var.numpy(), 0.0, None)) if sigma else None
        return Prediction(y=mean.numpy(), sigma=std)
