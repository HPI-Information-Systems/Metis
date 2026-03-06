from metis.profiling.cache import cached
from metis.utils.data_profiling.single_column.value_distribution.quartiles import (
    quartiles as _quartiles,
    interquartile_range as _interquartile_range,
)

quartiles = cached(_quartiles)
interquartile_range = cached(_interquartile_range)
