import numpy as np

from ezmodel.util.partitioning.crossvalidation import CrossvalidationPartitioning
from ezmodel.util.partitioning.random import RandomPartitioning


def test_random_selection():
    X = np.random.random((10000, 1))
    partitions = RandomPartitioning(perc_train=0.01, n_sets=5).do(X)

    assert len(partitions) == 5

    for train, test in partitions:
        assert len(train) == 100
        assert len(test) == 9900

        for i in train:
            assert i not in test


def test_crossvalidation_selection():
    X = np.random.random((100, 1))
    partitions = CrossvalidationPartitioning(k_folds=5, randomize=True).do(X)

    assert len(partitions) == 5
    for k, (train, test) in enumerate(partitions):

        assert len(train) == 80
        assert len(test) == 20

        for i in train:
            assert i not in test
