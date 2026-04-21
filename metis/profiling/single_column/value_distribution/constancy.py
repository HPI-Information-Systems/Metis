from metis.profiling.cache import cached
from metis.utils.data_profiling.single_column.value_distribution.constancy import (
    constancy as _constancy,
    most_frequent_value as _most_frequent_value,
)

constancy = cached(_constancy)
most_frequent_value = cached(_most_frequent_value)
