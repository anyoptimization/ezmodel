"""SMT-backed Kriging surrogate model (optional dependency)."""

import numpy as np

try:
    from smt.surrogate_models import KRG
except:  # noqa: E722  (optional dependency import guard)
    raise Exception("Model not found. Please execute: 'pip install smt'")

from ezmodel.core.model import Model
from ezmodel.core.prediction import Prediction


class smtKriging(Model):
    def __init__(self, theta0=1e-4, **kwargs) -> None:
        super().__init__(**kwargs)
        self.theta0 = theta0

    def _fit(self, X, y, **kwargs):
        n_var = X.shape[1]

        sm = KRG(theta0=[self.theta0] * n_var)
        sm.set_training_values(X, y)
        sm.train()

        self.model = sm

    def _predict(self, X, sigma=False, grad=False):
        y = self.model.predict_values(X)
        std = np.sqrt(np.clip(self.model.predict_variances(X), 0.0, None)) if sigma else None
        return Prediction(y=y, sigma=std)
