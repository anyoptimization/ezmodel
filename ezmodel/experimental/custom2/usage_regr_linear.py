import matplotlib.pyplot as plt
import numpy as np

from ezmodel.custom2.regr import LinearRegression
from ezmodel.util.sample_from_func import linear_function

# build the model to be used
model = LinearRegression()

# create some data to test this model on
X, y, _X, _y = linear_function(10000, 200, noise=1, n_var=1)

# let the model fit the data
model.fit(X, y)

# predict the data using the model
_X = _X[np.argsort(_X[:, 0])]
y_hat = model.predict(_X)

plt.scatter(X, y, label="Data")
plt.plot(_X, y_hat, color="black", label="Regression")
plt.legend()
plt.show()
