from metis.profiling.cache import cached
from metis.utils.data_profiling.single_column.cardinalities.uniqueness import (
    uniqueness as _uniqueness,
)

uniqueness = cached(_uniqueness)
