from metis.profiling.cache import cached
from metis.utils.data_profiling.single_column.cardinalities.value_length import (
    value_length_max as _value_length_max,
    value_length_mean as _value_length_mean,
    value_length_median as _value_length_median,
    value_length_min as _value_length_min,
)

value_length_max = cached(_value_length_max)
value_length_mean = cached(_value_length_mean)
value_length_median = cached(_value_length_median)
value_length_min = cached(_value_length_min)
