import numpy as np

from pymoo.util.misc import all_except
from ezmodel.custom.kernel import Linear, Quadratic, Constant, Sine
from ezmodel.custom.lsq import LSQ, LSQQR
from ezmodel.core.model import Model


def get_kernel(name):
    if name == "linear":
        return Linear()
    elif name == "quadratic":
        return Quadratic()
    elif name == "sine":
        return Sine()
    else:
        raise Exception("Unknown kernel.")


class Regression(Model):

    def __init__(self,
                 kernel="linear",
                 alpha=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.kernel = get_kernel(kernel)
        self.alpha = alpha
        self.lsq = None

    def _fit(self, X, y, weights=None, **kwargs):
        K = self.kernel.calc(X)
        self.lsq = LSQ(alpha=self.alpha)
        self.lsq.fit(K, y, weights=weights)

    def _predict(self, X, out, **kwargs):
        calc_gradient = "grad" in out

        if calc_gradient:
            H, grad_H = self.kernel.calc(X, Y=self.X, return_grad=True)
            out["y"] = self.lsq.predict(H)
            out["grad"] = grad_H @ self.lsq.beta
        else:
            H = self.kernel.calc(X, Y=self.X)
            out["y"] = self.lsq.predict(H)

    @classmethod
    def hyperparameters(cls):
        return dict(
            kernel=["linear", "quadratic", "sine"]
        )


class ConstantRegression(Regression):

    def __init__(self, **kwargs) -> None:
        super().__init__(Constant(), **kwargs)


class LinearRegression(Regression):

    def __init__(self, **kwargs) -> None:
        super().__init__(Linear(), **kwargs)


class QuadraticRegression(Regression):

    def __init__(self, **kwargs) -> None:
        super().__init__(Quadratic(), **kwargs)


class LocalRegression(Regression):

    def _fit(self, X, y, target=None, target_X=None, target_y=None, weights=None, **kwargs):
        if target is None:
            raise Exception("Please provide a target index which define the locality of the model!")

        if target is not None:
            j = target
            self._X = X[j]
            self._y = y[j]

            X = all_except(X, j) - self._X
            y = all_except(y, j) - self._y

            if weights is not None:
                weights = all_except(weights, j)

        elif target_X is not None and target_y is not None:
            self._X = target_X
            self._y = target_y
        else:
            raise Exception("Either provide target as index OR target_X and target_y!")

        super()._fit(X, y, weights=weights, **kwargs)

    def _predict(self, X, out, **kwargs):
        super()._predict(X - self._X, out, **kwargs)
        out["y"] = out["y"] + self._y


class LocalLinearRegression(LocalRegression):

    def __init__(self, **kwargs) -> None:
        super().__init__(Linear(no_intercept=True), **kwargs)


class LocalQuadraticRegression(LocalRegression):

    def __init__(self, **kwargs) -> None:
        super().__init__(Quadratic(no_intercept=True), **kwargs)


class RidgeRegression(Model):

    def __init__(self,
                 alpha=0.01,
                 alpha_opt=True,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        self.alpha = alpha
        self.alpha_opt = alpha_opt
        self.alpha_iter = 0

        self.v_e = None
        self.v_e_loo = None
        self.v_e_gcv = None

        self.e = None
        self.e_loo = None
        self.e_gcv = None

        self.model = None
        self.best_e_gcv = float("inf")

    def _optimize(self, **kwargs):

        X, y = self.X, self.y

        while True:

            H = Linear().calc(X)
            self.lsq = LSQQR(alpha=self.alpha, calc_A_inv=True, calc_proj=True)
            self.lsq.fit(H, y, weights=None)

            P, A_inv, w = self.lsq.P, self.lsq.A_inv, self.lsq.beta

            v_e = (H @ w - y)[:, 0]
            e = (v_e ** 2).sum()

            v_e_loo = v_e / np.diag(P)
            e_loo = (v_e_loo ** 2).mean()

            v_e_gcv = v_e / (np.trace(P) / len(P))
            e_gcv = (v_e_gcv ** 2).mean()

            self.e, self.e_loo, self.e_gcv = e, e_loo, e_gcv
            self.v_e, self.v_e_loo, self.v_e_gcv = v_e, v_e_loo, v_e_gcv

            if not self.alpha_opt or self.best_e_gcv - e_gcv < 1e-6:
                break

            self.best_e_gcv = e_gcv
            self.alpha = (e * np.trace(A_inv - self.alpha * A_inv ** 2)) / (w.T @ A_inv @ w * np.trace(P))
            self.alpha_iter += 1

    def _predict(self, X, out):
        H = Linear().calc(X, Y=self.X)
        out["y"] = self.lsq.predict(H)


def calc_min_samples(regr_or_kernel, n_var):
    kernel = regr_or_kernel.kernel if isinstance(regr_or_kernel, Regression) else regr_or_kernel

    if isinstance(kernel, Constant):
        n = 1

    elif isinstance(kernel, Linear):
        n = 1 + n_var

    elif isinstance(kernel, Quadratic):
        n = 1 + n_var + n_var * (n_var + 1) / 2

    return int(n)
