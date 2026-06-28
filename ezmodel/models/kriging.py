"""Kriging (Gaussian process) surrogate model."""

import numpy as np
from pydacefit.corr import RationalQuadratic
from pydacefit.dace import DACE
from pydacefit.regr import LinearRegression

from ezmodel.core.model import Model
from ezmodel.core.prediction import Prediction


class Kriging(Model):
    def __init__(self, regr=None, corr=None, ARD=False, theta=1.0, thetaL=0.00001, thetaU=100.0, **kwargs) -> None:

        super().__init__(eliminate_duplicates=True, **kwargs)
        # regr/corr are pydacefit objects (e.g. LinearRegression(), Gaussian(),
        # RationalQuadratic(alpha=0.25)) passed straight through to DACE. The default
        # kernel is RationalQuadratic(alpha=0.25) -- the best all-round performer across
        # the test-function benchmark; the trend defaults to a linear one.
        self.regr = regr if regr is not None else LinearRegression()
        self.corr = corr if corr is not None else RationalQuadratic(0.25)
        self.ARD = ARD
        self.theta = theta
        self.thetaL = thetaL
        self.thetaU = thetaU

    def _fit(self, X, y, **kwargs):
        theta, thetaL, thetaU = self.theta, self.thetaL, self.thetaU

        if self.ARD and self.thetaL is not None and self.thetaU is not None:
            _, m = X.shape
            theta = np.full(m, theta)
            thetaL = np.full(m, thetaL)
            thetaU = np.full(m, thetaU)

        self.model = DACE(regr=self.regr, corr=self.corr, theta=theta, thetaL=thetaL, thetaU=thetaU)
        self.model.fit(X, y)

    def _predict(self, X, sigma=False, grad=False):
        # DACE.predict returns its own Prediction(y, mse, grad); mse/grad are computed
        # only when requested and share the single Cholesky solve with the mean. The
        # gradient comes back in original (destandardized) space already.
        pred = self.model.predict(X, mse=sigma, grad=grad)
        std = np.sqrt(np.clip(pred.mse, 0.0, None)) if sigma else None
        return Prediction(y=pred.y, sigma=std, grad=pred.grad)
