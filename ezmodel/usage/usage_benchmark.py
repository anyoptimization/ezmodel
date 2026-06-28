"""Usage example: benchmark several surrogate models."""

import numpy as np
import pandas as pd

pd.set_option("display.expand_frame_repr", False)
pd.set_option("max_colwidth", 1000)

from pydacefit.corr import Gaussian, RationalQuadratic  # noqa: E402  (imports follow pandas display setup)

from ezmodel.core.benchmark import Benchmark  # noqa: E402  (imports follow pandas display setup)
from ezmodel.core.factory import cartesian  # noqa: E402  (imports follow pandas display setup)
from ezmodel.models.kriging import Kriging  # noqa: E402  (imports follow pandas display setup)
from ezmodel.models.rbf import RBF  # noqa: E402  (imports follow pandas display setup)
from ezmodel.util.partitioning.crossvalidation import CrossvalidationPartitioning  # noqa: E402

X = np.random.random((100, 3)) * 2 * np.pi
y = np.sin(X).sum(axis=1)

# build the models to compare as named grids (cartesian replaces the old hyperparameters())
models = {
    **cartesian(RBF, kernel=["cubic", "gaussian"], tail=["linear"]),
    **cartesian(Kriging, corr={"gauss": Gaussian(), "rq": RationalQuadratic()}),
}

# set up the benchmark and add the models to be used
benchmark = Benchmark(models, n_threads=4, verbose=True, raise_exception=True)

# create partitions to validate the performance of each model
partitions = CrossvalidationPartitioning(k_folds=5, seed=1).do(X)

# runs the experiment with the specified partitioning
benchmark.do(X, y, partitions=partitions)

# print out the benchmark results
print(benchmark.statistics("mae"))
