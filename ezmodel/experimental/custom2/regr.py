import numpy as np


class Regression:

    def __init__(self) -> None:
        super().__init__()
        self.beta = None

    def polynomial(self, X):
        pass

    def grad(self, X):
        pass

    def fit(self, X, y):
        P = self.polynomial(X)
        self.beta = np.linalg.lstsq(P, y, rcond=None)[0]

    def predict(self, X):
        P = self.polynomial(X)
        return P @ self.beta


class ConstantRegression(Regression):

    def fit(self, X, y):
        self.beta = np.array([y.mean()])

    def polynomial(self, X):
        return np.ones((len(X), 1))

    def grad(self, X):
        return self.polynomial(X)


class LinearRegression(Regression):

    def polynomial(self, X):
        return np.column_stack((np.ones(len(X)), X))

    def grad(self, X):
        return self.polynomial(X)


class QuadraticRegression(Regression):

    def polynomial(self, X):
        return np.column_stack((np.ones(len(X)), X ** 2))

    def grad(self, X):
        return np.column_stack((np.ones(len(X)), 2 * X))


class LinearAndQuadraticRegression(Regression):

    def polynomial(self, X):
        return np.column_stack((np.ones(len(X)), X, X ** 2))

    def grad(self, X):
        return np.column_stack((np.ones(len(X)), X, 2 * X))


REGR = {
    "constant": ConstantRegression,
    "linear": LinearRegression,
    "quadratic": QuadraticRegression,
    "linear+quadratic": LinearAndQuadraticRegression,
}


def get_regr(regr):
    clazz = REGR.get(regr)
    if clazz is not None:
        return clazz()


"""


    def fit(self, X, y):
        P = self.polynomial(X)

        optim = self.optim
        if optim == "auto":
            if len(X) > 50:
                optim = "adam"
            else:
                optim = "lstsq"

        if optim == "lstsq":
            self.beta = np.linalg.lstsq(P, y, rcond=None)[0]
        else:

            n_max_iter = 10000
            batch_size = 20
            lr = 0.0001
            tol = 1e-6

            G = self.grad(X)
            _, n = P.shape
            beta = np.random.random((n, 1)) * 2 - 1

            for k in range(n_max_iter):

                _beta = beta

                n_batches = len(X) // batch_size

                for j in range(n_batches):
                    I = np.random.randint(0, len(X), batch_size)

                    _P = P[I]
                    _X = X[I]
                    _y = y[I]

                    y_hat = _P @ beta
                    y_hat_deriv_by_beta = G[I]

                    mse_by_beta = 2 * (y_hat - _y) * y_hat_deriv_by_beta

                    grad = mse_by_beta.sum(axis=0)

                    beta = beta - lr * grad[:, None]

                mse = ((P @ beta - y) ** 2).sum()

                print(k, mse)

                if np.all(np.abs(beta - _beta) < tol):
                    break



"""
