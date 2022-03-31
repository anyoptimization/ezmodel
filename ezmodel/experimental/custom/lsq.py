import numpy as np
import scipy

from ezmodel.core.model import Model


class LSQ:

    def __init__(self, alpha=None):
        super().__init__()
        self.alpha = alpha
        self.beta = None

    def fit(self, H, y, weights=None, **kwargs):
        _H, _y = H, y

        if weights is not None:
            _W = np.eye(len(H)) * (weights ** 0.5)
            _H, _y = _W @ _H, _W @ _y

        if self.alpha is not None:
            _, m = _H.shape
            _H = np.row_stack([_H, np.sqrt(self.alpha) * np.eye(m)])
            _y = np.row_stack([_y, np.zeros((m, 1))])

        self.beta = np.linalg.lstsq(_H, _y, rcond=None)[0]

    def predict(self, H):
        return H @ self.beta


class LSQQR(LSQ):

    def __init__(self,
                 alpha=None,
                 calc_A_inv=False,
                 calc_proj=False):

        super().__init__()
        self.alpha = alpha
        self.K = None
        self.Q = None
        self.R = None
        self.P = None

        self.beta = None

        self.calc_A_inv = calc_A_inv or calc_proj
        self.calc_proj = calc_proj

        self.A = None
        self.A_inv = None

    def fit(self, H, y, weights=None, **kwargs):
        _H, _y = H, y

        if weights is not None:
            _W = np.eye(len(H)) * (weights ** 0.5)
            _H, _y = _W @ _H, _W @ _y

        Q, R = np.linalg.qr(_H, mode='reduced')

        if self.alpha is None:
            _A = R
            _rhs = Q.T
            beta = np.linalg.lstsq(_H, _y, rcond=None)[0]

            if self.calc_A_inv:
                self.A_inv, _ = scipy.linalg.lapack.dtrtri(R)

        else:
            _A = R.T @ R + np.eye(len(R)) * self.alpha
            _rhs = R.T @ Q.T
            beta = np.linalg.solve(_A, _rhs @ y)

            if self.calc_A_inv:
                self.A_inv = np.linalg.inv(_A)

        self.H, self.Q, self.R, self.beta = H, Q, R, beta
        self.A, self.rhs, self._y = _A, _rhs, _y

        if self.calc_proj:
            self.P = np.eye(len(H)) - H @ self.A_inv @ H.T

    def predict(self, H):
        return H @ self.beta


class LSQInv(Model):

    def __init__(self, alpha=0.0):
        super().__init__()
        self.alpha = alpha

        self.H = None
        self.A = None
        self.A_inv = None
        self.beta = None

    def _fit(self, H, y, weights=None, **kwargs):

        W = np.eye(len(H))
        if weights is not None:
            W = W * weights

        A = H.T @ W @ H + np.eye(H.shape[1]) * self.alpha

        try:
            A_inv = np.linalg.inv(A)
            self.singular = False
        except:
            A_inv = np.linalg.pinv(A)
            self.singular = True

        beta = A_inv @ H.T @ W @ y
        self.H, self.A, self.A_inv, self.beta = H, A, A_inv, beta
        self.P = np.eye(len(H)) - H @ A_inv @ H.T

    def _predict(self, H):
        return H @ self.beta
