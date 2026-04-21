from metis.profiling.cache import cached
from metis.utils.data_profiling.single_column.patterns_and_data_types.numeric_precision import (
    size as _size,
    decimals as _decimals,
)

size = cached(_size)
decimals = cached(_decimals)
