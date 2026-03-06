from metis.profiling.cache import cached
from metis.utils.data_profiling.single_column.patterns_and_data_types.patterns import (
    patterns as _patterns,
)

patterns = cached(_patterns)
