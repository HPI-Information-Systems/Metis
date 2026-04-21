from metis.profiling.cache import cached
from metis.utils.data_profiling.single_column.cardinalities.row_count import (
    row_count as _row_count,
)

row_count = cached(_row_count)
