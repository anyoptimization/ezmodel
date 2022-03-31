import autograd.numpy as anp
import numpy as np

from ezmodel.custom2.regr import get_regr
from ezmodel.custom2.util import eucl_dist


class KernelFactory:

    def __init__(self, clazz, norm=True, theta=None, eps=1e-6, tail="constant", func_dist=eucl_dist, **kwargs) -> None:
        super().__init__()
        if isinstance(clazz, str):
            clazz = get_kernel(clazz)

        self.clazz = clazz
        self.kwargs = dict(norm=norm, theta=theta, eps=eps, tail=tail, func_dist=func_dist)

    def create(self, **kwargs):
        params = dict(self.kwargs)
        for k, v in kwargs.items():
            params[k] = v
        return self.clazz(**params)


class Kernel:

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.X = None
        self.D_max = None
        self.beta = None
        self.mse = None

    def from_dict(self, kwargs, key):
        return kwargs.get(key, self.kwargs.get(key))

    def D(self, X, Z, **kwargs):
        theta = self.from_dict(kwargs, "theta")
        func_dist = self.from_dict(kwargs, "func_dist")

        # prepare the left and right for distances
        u = np.repeat(X, Z.shape[0], axis=0)
        v = np.tile(Z, (X.shape[0], 1))

        # calculate the distances and reshape to distance metric
        D = func_dist(u, v)
        D = np.reshape(D, (X.shape[0], Z.shape[0]))

        # if a hyper-parameter theta should be considered
        if theta is not None:
            D = D / (theta ** 2)

        return D

    def R(self, X, Z=None, **kwargs):
        eps = self.from_dict(kwargs, "eps")
        tail = self.from_dict(kwargs, "tail")
        norm = self.from_dict(kwargs, "norm")
        R_max = self.from_dict(kwargs, "R_max")

        state = {}

        # defines whether the kernel is calculate to the X itself or to another set Z
        only_X = False

        # if Z is not provided calculate the kernel to itself
        if Z is None:
            only_X = True
            Z = X

        # calculate the distances between X and Z - ignore theta for now
        D = self.D(X, Z, **kwargs)

        # now use the actual kernel implementation
        R = self._R(D, **kwargs)

        # if the kernel matrix should be normalized
        if norm:

            # find the normalization constant if not provided
            if R_max is None:
                R_max = D.sum(axis=0)[None, :]
                state["R_max"] = R_max

            # now include the norm of each row in the matrix
            R = R / R_max

        # check whether an epsilon should be added to the diagonal
        if only_X and eps is not None:
            R = R + (eps ** 2) * anp.eye(len(R))

        # if we should consider a tail do that
        if tail is not None:
            regr = get_regr(tail)
            P = regr.polynomial(X)
            _, d = P.shape
            R = anp.column_stack([R, P])

            # make sure it is a square matrix
            if only_X:
                R = anp.row_stack([R, np.column_stack([P.T, np.zeros((d, d))])])

        return R, state

    def fit(self, X, Y, **kwargs):
        _, d = X.shape

        # write the provided settings to the object before fitting
        for k, v in kwargs.items():
            self.kwargs[k] = v

        # build the kernel
        R, state = self.R(X, **kwargs)

        # extend the kernel by other state attributes returned by R
        self.kwargs = {**self.kwargs, **state}

        try:
            R_inv, cond = svd_inv(R)
        except Exception as e:
            raise e

        # initialize the right hand side to find the coefficients
        rhs = np.zeros((len(R), 1))
        rhs[:len(Y)] = Y

        # calculate the prediction coefficients of the kernel
        beta = R_inv @ rhs

        # an easy way to find the cross-validation error of the model
        if cond < 1e+19:
            c, _R = beta[:d, 0], R_inv[:d, :d]
            e = c / (anp.diag(_R) + 1e-128)
            mse = (e ** 2).sum()
        else:
            mse = np.inf

        self.X = X
        self.beta = beta
        self.mse = mse

    def predict(self, X):
        assert self.beta is not None, "You have to fit the kernel first before predicting"
        assert X.ndim == 2

        out = {}

        _R, _ = self.R(X, self.X)
        y_hat = _R @ self.beta
        out["y"] = y_hat

        return out

    def _R(self, D, **kwargs):
        pass

    def params(self):
        pass


class GaussianKernel(Kernel):

    def _R(self, D, **kwargs):
        return anp.exp(-0.5 * D)


class CubicKernel(Kernel):

    def _R(self, D, **kwargs):
        return (D ** 1.5)


class LinearKernel(Kernel):

    def _R(self, D, **kwargs):
        return anp.sqrt(D+1e-128)



KERNEL = {
    "gaussian": GaussianKernel,
    "cubic": CubicKernel,
    "linear": LinearKernel,
}


def get_kernel(kernel):
    clazz = KERNEL.get(kernel)
    if clazz is not None:
        return clazz


def svd_inv(A):
    # return anp.linalg.inv(A), None
    U, S, V = anp.linalg.svd(A, full_matrices=False)
    Ainv = V.T @ anp.diag(1 / S) @ U.T
    cond = anp.abs(anp.max(S)) / anp.abs(anp.min(S))
    return Ainv, cond
