"""Base ``Model`` class: the fit/predict lifecycle with pre- and post-processing."""

import time

import numpy as np

from ezmodel.core.prediction import Prediction
from ezmodel.core.transformation import NoNormalization
from ezmodel.util.misc import at_least2d, is_duplicate


class Model:
    def __init__(
        self,
        norm_X=NoNormalization(),
        norm_y=NoNormalization(),
        active_dims=None,
        filter_nan_and_inf=True,
        eliminate_duplicates=False,
        eliminate_duplicates_eps=1e-16,
        raise_exception_while_fitting=True,
        raise_exception_while_prediction=True,
        verbose=False,
        **kwargs,
    ):

        self.norm_X = norm_X
        self.norm_y = norm_y

        self.eliminate_duplicates = eliminate_duplicates
        self.eliminate_duplicates_eps = eliminate_duplicates_eps

        self.active_dims = active_dims
        self.filter_nan_and_inf = filter_nan_and_inf
        self.verbose = verbose

        self.time = None
        self.model = None
        self.X, self.y = None, None
        self._X, self._y = None, None
        self.success = None
        self.data = {}

        self.raise_exception_while_fitting = raise_exception_while_fitting
        self.raise_exception_while_prediction = raise_exception_while_prediction
        self.exception = None

        self.has_been_fitted = False

    def preprocess(self, X, y, **kwargs):

        if self.active_dims is not None:
            X = X[:, self.active_dims]

        if self.eliminate_duplicates:
            I = ~is_duplicate(X, eps=self.eliminate_duplicates_eps)  # noqa: E741  (I is an identity matrix)
            X, y = X[I], y[I]

        if self.filter_nan_and_inf:
            X_I = np.all(~np.isnan(X) & ~np.isinf(X), axis=1)
            y_I = np.all(~np.isnan(y) & ~np.isinf(y), axis=1)
            X, y = X[X_I & y_I], y[X_I & y_I]

        X, y = self.norm_X.forward(X), self.norm_y.forward(y)
        X, y = self._preprocess(X, y, **kwargs)

        return X, y

    def postprocess(self, pred):
        pred = Prediction(y=self.norm_y.backward(pred.y), sigma=pred.sigma, grad=pred.grad)
        return self._postprocess(pred)

    def fit(self, X, y, **kwargs):
        X, y = at_least2d(X, expand="r"), at_least2d(y, expand="c")
        assert len(X) == len(y)
        self._X, self._y = X, y

        self.X, self.y = self.preprocess(X, y)

        start = time.time()

        try:
            # fit the model given the data
            self._fit(self.X, self.y, **kwargs)

            # do some parameter optimization if the model requires it
            self._optimize(**kwargs)

            # if no exception occurs set the model to be fitted successfully
            self.success = True

        except Exception as ex:
            self.success = False
            self.exception = ex
            if self.raise_exception_while_fitting:
                raise ex

        self.has_been_fitted = True
        self.time = time.time() - start

        return self

    def predict(self, X, sigma=False, grad=False):
        """Predict the mean (and optionally standard deviation/gradient) for ``X``.

        Args:
            X: Query points, shape ``(m, d)``.
            sigma: Also return the predictive standard deviation (``Prediction.sigma``).
            grad: Also return the gradient of the mean w.r.t. ``X`` (``Prediction.grad``).

        Returns:
            A :class:`~ezmodel.core.prediction.Prediction` whose ``y`` is always set;
            ``sigma``/``grad`` are populated only when their flag is requested (and the
            model supports them, else ``None``).

        Note:
            ``sigma`` and ``grad`` are returned in the model's output space and are only
            un-normalized for the default ``norm_y=NoNormalization`` (which every
            uncertainty-providing model uses). With a non-identity ``norm_y`` only ``y``
            is back-transformed; ``sigma``/``grad`` would need its scale/Jacobian.
        """
        q = self._y.shape[1] if self._y is not None else 1

        if not self.success:
            if self.raise_exception_while_fitting:
                raise Exception("There was an error while fitting the model.")
            else:
                return Prediction(y=np.full((len(X), q), np.nan))

        Xq = X[:, self.active_dims] if self.active_dims is not None else X
        Xq = self.norm_X.forward(at_least2d(Xq, expand="r"))

        try:
            pred = self._predict(Xq, sigma=sigma, grad=grad)
            pred = self.postprocess(pred)
        except Exception as e:
            if self.raise_exception_while_prediction:
                raise e
            else:
                pred = Prediction(y=np.full((len(X), q), np.inf))

        return pred

    def _preprocess(self, X, y, **kwargs):
        return X, y

    def _postprocess(self, pred):
        return pred

    def _fit(self, X, y, **kwargs):
        pass

    def _predict(self, X, sigma=False, grad=False):
        raise NotImplementedError

    def _optimize(self, **kwargs):
        pass
