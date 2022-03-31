from ezmodel.core.factory import ModelFactoryByClazz
from ezmodel.core.partitioning import merge_and_partition
from ezmodel.core.selection import ModelSelection
from ezmodel.models.kriging import Kriging
from ezmodel.util.sample_from_func import sine_function


def test_selection():
    X, y, _X, _y = sine_function(100, 20)

    models = ModelFactoryByClazz(Kriging).do()
    X, y, partitions = merge_and_partition((X, y), (_X, _y))

    model = ModelSelection(models, refit=False).do(X, y, partitions=partitions)

    assert model is not None

