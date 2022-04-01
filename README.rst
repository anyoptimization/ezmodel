ezmodel - A Common Interface for Models and Model Selection
====================================================================

For more information about our toolbox, users are encouraged to read our documentation.
https://anyoptimization.com/projects/ezmodel/


|python| |license|


.. |python| image:: https://img.shields.io/badge/python-3.9-blue.svg
   :alt: python 3.6

.. |license| image:: https://img.shields.io/badge/license-apache-orange.svg
   :alt: license apache
   :target: https://www.apache.org/licenses/LICENSE-2.0



Installation
====================================================================

The official release is always available at PyPi:

.. code:: bash

    pip install -U ezmodel



Usage
==================================



Benchmarking
==================================


.. code:: python

    
    import numpy as np

    import pandas as pd
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', 1000)

    from ezmodel.core.benchmark import Benchmark
    from ezmodel.core.factory import models_from_clazzes
    from ezmodel.models.kriging import Kriging
    from ezmodel.models.rbf import RBF
    from ezmodel.util.partitioning.crossvalidation import CrossvalidationPartitioning

    X = np.random.random((100, 3)) * 2 * np.pi
    y = np.sin(X).sum(axis=1)

    models = models_from_clazzes(RBF, Kriging)

    # set up the benchmark and add the models to be used
    benchmark = Benchmark(models, n_threads=4, verbose=True, raise_exception=True)

    # create partitions to validate the performance of each model
    partitions = CrossvalidationPartitioning(k_folds=5, seed=1).do(X)

    # runs the experiment with the specified partitioning
    benchmark.do(X, y, partitions=partitions)

    # print out the benchmark results
    print(benchmark.statistics("mae"))



::

                                                                      mae
                                                                     mean       std       min        max    median
    label
    Kriging[regr=constant,corr=gauss,thetaU=100,ARD=False]       0.017159  0.007472  0.009658   0.025359  0.014855
    Kriging[regr=constant,corr=gauss,thetaU=20,ARD=False]        0.017159  0.007472  0.009658   0.025359  0.014855
    Kriging[regr=linear,corr=gauss,thetaU=100,ARD=False]         0.018064  0.008069  0.010350   0.027456  0.014246
    Kriging[regr=linear,corr=gauss,thetaU=20,ARD=False]          0.018064  0.008069  0.010350   0.027456  0.014246
    Kriging[regr=constant,corr=gauss,thetaU=100,ARD=True]        0.021755  0.007409  0.011955   0.028896  0.025163
    Kriging[regr=constant,corr=gauss,thetaU=20,ARD=True]         0.021755  0.007409  0.011955   0.028896  0.025163
    Kriging[regr=linear,corr=gauss,thetaU=20,ARD=True]           0.025018  0.011348  0.011576   0.040585  0.022124
    Kriging[regr=linear,corr=gauss,thetaU=100,ARD=True]          0.025018  0.011348  0.011576   0.040585  0.022124
    Kriging[regr=constant,corr=exp,thetaU=100,ARD=False]         0.034493  0.009328  0.025092   0.045610  0.030661
    Kriging[regr=constant,corr=exp,thetaU=20,ARD=False]          0.034493  0.009328  0.025092   0.045610  0.030661
    Kriging[regr=linear,corr=exp,thetaU=100,ARD=False]           0.035734  0.009922  0.025611   0.047926  0.031473
    Kriging[regr=linear,corr=exp,thetaU=20,ARD=False]            0.035734  0.009922  0.025611   0.047926  0.031473
    Kriging[regr=constant,corr=exp,thetaU=100,ARD=True]          0.051527  0.010941  0.037944   0.065866  0.047440
    Kriging[regr=constant,corr=exp,thetaU=20,ARD=True]           0.051527  0.010941  0.037944   0.065866  0.047440
    Kriging[regr=linear,corr=exp,thetaU=100,ARD=True]            0.065867  0.025312  0.039058   0.104449  0.059957
    Kriging[regr=linear,corr=exp,thetaU=20,ARD=True]             0.065867  0.025312  0.039058   0.104449  0.059957
    RBF[kernel=cubic,tail=quadratic,normalized=True]             0.121947  0.033552  0.077895   0.167120  0.127345
    RBF[kernel=cubic,tail=constant,normalized=True]              0.125348  0.037982  0.072579   0.169413  0.140753
    RBF[kernel=cubic,tail=linear,normalized=True]                0.125474  0.038609  0.071268   0.169843  0.137987
    RBF[kernel=cubic,tail=linear+quadratic,normalized=True]      0.126070  0.039773  0.071279   0.171862  0.135489




RBF
----------------------------------


.. code:: python

    
    import matplotlib.pyplot as plt
    import numpy as np

    from ezmodel.models.rbf import RBF
    from ezmodel.util.sample_from_func import sine_function

    rbf = RBF(kernel="gaussian")

    # create some data to test this model on
    X, y, _X, _y = sine_function(20, 200)

    # let the model fit the data
    rbf.fit(X, y)

    # predict the data using the model
    y_hat = rbf.predict(_X)

    # predict the data using the model
    _X = _X[np.argsort(_X[:, 0])]
    y_hat = rbf.predict(_X)

    plt.scatter(X, y, label="Data")
    plt.plot(_X, y_hat, color="black", label="RBF")
    plt.legend()
    plt.show()



Kriging
----------------------------------


.. code:: python

    
    import matplotlib.pyplot as plt
    import numpy as np

    from ezmodel.models.kriging import Kriging
    from ezmodel.util.sample_from_func import square_function

    model = Kriging(regr="linear",
                    corr="gauss",
                    ARD=False)

    # create some data to test this model on
    X, y, _X, _y = square_function(100, 20)

    # let the model fit the data
    model.fit(X, y)

    # predict the data using the model
    y_hat = model.predict(_X)

    # predict the data using the model
    _X = _X[np.argsort(_X[:, 0])]
    y_hat = model.predict(_X)

    plt.scatter(X, y, label="Data")
    plt.plot(_X, y_hat, color="black", label="RBF")
    plt.legend()
    plt.show()




Contact
=======


Feel free to contact us if you have any question:

::

    Julian Blank (blankjul [at] msu.edu)
    Michigan State University
    Computational Optimization and Innovation Laboratory (COIN)
    East Lansing, MI 48824, USA
