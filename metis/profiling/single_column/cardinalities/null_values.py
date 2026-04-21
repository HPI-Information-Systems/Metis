from metis.profiling.cache import cached
from metis.utils.data_profiling.single_column.cardinalities.null_values import (
    null_count as _null_count,
    null_percentage as _null_percentage,
)

null_count = cached(_null_count)
null_percentage = cached(_null_percentage)
