import numpy as np


def squared_dist(A, B=None, theta=None):
    if B is None:
        B = A
    diff = (A[:, None] - B[None, :])
    if theta is not None:
        diff *= theta
    return (diff ** 2).sum(axis=2)


def dist(A, B=None, theta=None):
    return squared_dist(A, B=B, theta=theta) ** 0.5


class Kernel:

    def __init__(self, sigma=1.0, theta=None) -> None:
        super().__init__()
        self.theta = theta
        self.sigma = sigma

    def calc(self, X, Y=None):
        pass


# --------------------------------------------------------------------
# Radial
# --------------------------------------------------------------------


class Gaussian(Kernel):

    def calc(self, X, Y=None):
        D = squared_dist(X, Y, theta=self.theta)
        return np.exp(- 0.5 * (D / self.sigma))


# --------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------


class Constant(Kernel):

    def calc(self, X, return_grad=False, **kwargs):
        n, _ = X.shape
        ret = np.ones((n, 1))

        if return_grad:
            grad = np.column_stack([np.zeros((X.shape[1], 1))])
            return ret, grad
        else:
            return ret


class Linear(Kernel):

    def __init__(self, no_intercept=False) -> None:
        super().__init__()
        self.no_intercept = no_intercept

    def calc(self, X, return_grad=False, **kwargs):
        n, _ = X.shape
        ret = np.column_stack([np.ones(n), X]) if not self.no_intercept else X

        if return_grad:
            grad = np.column_stack([np.zeros((X.shape[1], 1)), np.eye(X.shape[1])])
            return ret, grad
        else:
            return ret




class Quadratic(Kernel):

    def __init__(self, no_intercept=False) -> None:
        super().__init__()
        self.no_intercept = no_intercept

    def calc(self, X, return_grad=False, **kwargs):
        n, m = X.shape
        M = np.column_stack([X[:, [k]] * X[:, range(k, m)] for k in range(m)])

        ret = np.column_stack([X, M])
        if not self.no_intercept:
            ret = np.column_stack([np.ones((n, 1)), ret])

        if return_grad:
            raise Exception("Not Implemented yet.")

        return ret

        """
        nn = int((n + 1) * (n + 2) / 2)

        grad = np.column_stack([np.zeros((n, 1)), np.eye(n), np.zeros((n, nn - n - 1))])
        q = n
        j = n + 1

        for k in range(n):
            grad[k, j + np.arange(q)] = np.column_stack([2 * X[:, [k]], X[:, np.arange(k + 1, n)]])
            for i in range(n - k - 1):
                grad[k + i + 1, j + 1 + i] = X[0, k]
            j = j + q
            q = q - 1

        return ret, grad
        """


class Sine(Kernel):

    def __init__(self, no_intercept=False) -> None:
        super().__init__()
        self.no_intercept = no_intercept

    def calc(self, X, return_grad=False, **kwargs):
        n, _ = X.shape
        ret = np.column_stack([X, X ** 2, np.sin(np.pi * X)])
        if not self.no_intercept:
            ret = np.column_stack([np.ones(n), ret])

        if return_grad:
            raise Exception("Not implemented yet!")
        else:
            return ret
