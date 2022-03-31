from ezmodel.core.benchmark import Benchmark
from ezmodel.core.factory import ModelFactoryByClazz
from ezmodel.models.kriging import Kriging
from ezmodel.util.partitioning.crossvalidation import CrossvalidationPartitioning
from ezmodel.util.sample_from_func import sine_function


def test_benchmark():
    X, y, _X, _y = sine_function(100, 20)

    models = ModelFactoryByClazz(Kriging).do()
    benchmark = Benchmark(models)

    partitions = CrossvalidationPartitioning(5).do(X)
    benchmark.do(X, y, partitions)

    vals = benchmark.statistics()

    assert vals is not None


