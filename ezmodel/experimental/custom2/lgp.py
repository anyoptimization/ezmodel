import autograd.numpy as anp
import numpy as np
from autograd import grad

from ezmodel.custom2.kernel import GaussianKernel, KernelFactory
from ezmodel.custom2.optimizer import Adam
from ezmodel.core.model import Model
from ezmodel.core.transformation import NoNormalization
from ezmodel.util.transformation.zero_to_one import ZeroToOneNormalization


def predict(params, fac, n_closest, X, Y, x):
    kernel = kernel_from_params(fac, params)

    d = kernel.D(x[None, :], X)[0]
    closest = d.argsort()[:n_closest]

    kernel.fit(X[closest], Y[closest])

    out = kernel.predict(x[None, :])
    y_hat = out["y"][0]

    return y_hat, kernel


def mse(params, fac, n_closest, X, Y, x, y):
    y_hat, kernel = predict(params, fac, n_closest, X, Y, x)
    mse = (y_hat - y) ** 2
    return mse[0]


def kernel_from_params(fac, params):
    theta = anp.exp(params[0])
    kernel = fac.create(theta=theta)
    return kernel


def avg_mse(params, X, y, n_nearest, fac):
    _mse = []
    for j in range(len(X)):
        m = np.full(len(X), True)
        m[j] = False

        __mse = mse(params, fac, n_nearest, X[m], y[m], X[j], y[j])
        _mse.append(__mse)

    return np.array(_mse).mean()


class LGP(Model):

    def __init__(self,
                 n_nearest=None,
                 norm_X=NoNormalization(),
                 # norm_y=NoNormalization(),
                 kernel="gaussian",
                 n_max_iter=0,
                 # norm_X=Standardization(),
                 norm_y=ZeroToOneNormalization(),
                 verbose=False,
                 **kwargs):

        super().__init__(norm_X=norm_X, norm_y=norm_y, **kwargs)
        self.n_nearest = n_nearest
        self.n_max_iter = n_max_iter
        self.fac = KernelFactory(kernel, **kwargs)
        self.verbose = verbose

    def _fit(self, X, y, **kwargs):

        n_nearest = self.n_nearest

        if n_nearest is None:
            params = np.array([anp.log(1.0)])

            K = np.arange(3, 16).astype(int)
            vals = np.array([avg_mse(params, X, y, k, self.fac) for k in K])
            n_nearest = K[vals.argmin()]

            self.n_nearest = n_nearest

        thetas = np.linspace(0.01, 10, 50)
        vals = np.array([avg_mse(np.array([anp.log(theta)]), X, y, n_nearest, self.fac) for theta in thetas])
        theta = thetas[vals.argmin()]

        params = np.array([anp.log(theta)])

        optim = Adam(params)
        # optim = SGD(params, alpha=0.0001)

        for i in range(self.n_max_iter):

            _grad = []

            # for j in np.random.permutation(len(X)):
            for j in range(len(X)):

                m = np.full(len(X), True)
                m[j] = False

                # _mse = mse(optim.X, self.fac, n_nearest, X[m], y[m], X[j], y[j])

                func_grad = grad(mse)
                __grad = func_grad(optim.X, self.fac, n_nearest, X[m], y[m], X[j], y[j])

                _grad.append(__grad)

                if np.any(np.isnan(__grad)):
                    func_grad(optim.X, self.fac, n_nearest, X[m], y[m], X[j], y[j])

            if self.verbose:
                _mse = avg_mse(optim.X, X, y, n_nearest, self.fac)
                print(i, np.exp(optim.X), _mse)

            _grad = np.array(_grad).sum(axis=0)

            if np.all(_grad < 1e-8) or np.any(np.isnan(_grad)):
                break

            optim.apply(_grad)

        # print(np.array(_mse).mean())
        self.params = optim.X

    def _predict(self, X, out):

        y = []
        for j in range(len(X)):
            y_hat, _ = predict(self.params, self.fac, self.n_nearest, self.X, self.y, X[j])
            y.append(y_hat)

        out["y"] = np.array(y)

    @classmethod
    def hyperparameters(cls):
        return {
            "kernel": ["gaussian"],
            "tail": ["constant", "linear"],
            "norm": [True, False],
        }
