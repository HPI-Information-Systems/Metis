from metis.profiling.cache import cached
from metis.utils.data_profiling.single_column.cardinalities.distinct_values import (
    distinct_count as _distinct_count,
)

distinct_count = cached(_distinct_count)
