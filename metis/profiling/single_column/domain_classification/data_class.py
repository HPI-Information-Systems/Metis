from metis.profiling.cache import cached
from metis.utils.data_profiling.single_column.domain_classification.data_class import (
    data_class as _data_class,
)

data_class = cached(_data_class)
