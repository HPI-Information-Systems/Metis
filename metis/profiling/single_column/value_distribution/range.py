from metis.profiling.cache import cached
from metis.utils.data_profiling.single_column.value_distribution.range import (
    value_range as _value_range,
)

value_range = cached(_value_range)
