from ezmodel.custom2.kernel import GaussianKernel
from ezmodel.util.sample_from_func import linear_function


# create some data to test this model on
X, y, _X, _y = linear_function(50, 200, noise=1, n_var=1)


kernel = GaussianKernel(tail="linear")

R = kernel.R(X)

