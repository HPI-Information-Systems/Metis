from metis.profiling.cache import cached
from metis.utils.data_profiling.single_column.patterns_and_data_types.data_type import (
    data_type as _data_type,
)

data_type = cached(_data_type)
