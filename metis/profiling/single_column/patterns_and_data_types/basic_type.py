from metis.profiling.cache import cached
from metis.utils.data_profiling.single_column.patterns_and_data_types.basic_type import (
    basic_type as _basic_type,
)

basic_type = cached(_basic_type)
