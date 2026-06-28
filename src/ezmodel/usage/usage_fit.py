"""Usage example: fit and select the best surrogate via ``fit``."""

import matplotlib.pyplot as plt
import numpy as np
from pydacefit.corr import Gaussian, RationalQuadratic

from ezmodel.core.factory import cartesian
from ezmodel.core.selection import ModelSelection  # noqa: F401  (public surface; used downstream e.g. pysamoo)
from ezmodel.fit import fit
from ezmodel.models.kriging import Kriging
from ezmodel.models.rbf import RBF
from ezmodel.util.partitioning.random import RandomPartitioning
from ezmodel.util.sample_from_func import square_function

X, y, _X, _y = square_function(20, 200, n_var=1)

# build the candidate models as a named grid (cartesian replaces the old hyperparameters())
models = {
    **cartesian(Kriging, corr={"gauss": Gaussian(), "rq": RationalQuadratic()}),
    **cartesian(RBF, kernel=["cubic", "gaussian"], tail=["linear"]),
}

# select the best of those models on the data
model = fit(X, y, models=models, partitions=RandomPartitioning().do(X))

# predict over a sorted grid for a clean line plot
_X = _X[np.argsort(_X[:, 0])]
y_hat = model.predict(_X).y

plt.scatter(X, y, label="Data")
plt.plot(_X, y_hat, color="black", label="Regression")
plt.legend()
plt.show()
