"""GPflow-backed Gaussian process surrogate model (optional dependency)."""

from ezmodel.core.model import Model

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

    def _predict(self, X, out, **kwargs):
        mean, var = self.model.predict_f(X)

        out["y"] = mean.numpy()
        if "var" in out:
            out["var"] = var.numpy()
