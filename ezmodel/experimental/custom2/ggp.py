import autograd.numpy as anp
import numpy as np
from autograd import value_and_grad

from ezmodel.custom2.kernel import KernelFactory
from ezmodel.custom2.optimizer import Adam
from ezmodel.core.model import Model
from ezmodel.core.transformation import NoNormalization


def mse(params, fac, X, Y):
    kernel = kernel_from_params(fac, params)
    kernel.fit(X, Y)
    return kernel.mse


def kernel_from_params(fac, params):
    theta = anp.exp(params[0])
    kernel = fac.create(theta=theta)
    return kernel


class GGP(Model):

    def __init__(self,
                 norm_X=NoNormalization(),
                 norm_y=NoNormalization(),
                 norm=False,
                 kernel="gaussian",
                 n_max_iter=10,
                 verbose=False,
                 # norm_X=Standardization(),
                 # norm_y=Standardization(),
                 **kwargs):

        super().__init__(eliminate_duplicates=True, eliminate_duplicates_eps=1e-8, norm_X=norm_X, norm_y=norm_y,
                         **kwargs)
        self.fac = KernelFactory(kernel, norm=norm, **kwargs)
        self.kernel = None
        self.n_max_iter = n_max_iter
        self.verbose = verbose

    def _fit(self, X, y, **kwargs):

        # thetas = np.linspace(0.01, 10, 50)
        # vals = np.array([mse(np.array([anp.log(theta)]), self.fac, X, y) for theta in thetas])
        # theta = thetas[vals.argmin()]
        # params = np.array([anp.log(theta)])

        params = np.array([anp.log(1.0)])

        optim = Adam(params)

        for i in range(self.n_max_iter):
            _mse, _grad = value_and_grad(mse)(optim.X, self.fac, X, y)

            if self.verbose:
                print(i, np.exp(optim.X), _mse)

            if np.all(_grad < 1e-8) or np.any(np.isnan(_grad)):
                break

            optim.apply(_grad)

        kernel = kernel_from_params(self.fac, optim.X)
        kernel.fit(X, y)
        self.kernel = kernel

    def _predict(self, X, out):
        ret = self.kernel.predict(X)
        for k, v in ret.items():
            out[k] = v

    @classmethod
    def hyperparameters(cls):
        return {
            "kernel": ["gaussian", "cubic", "linear"],
            "tail": ["none", "constant", "linear", "quadratic", "linear+quadratic"],
            "norm": [False, True]
        }
