from metis.profiling.cache import cached
from metis.utils.data_profiling.single_column.value_distribution.histogram import (
    equi_width_histogram as _equi_width_histogram,
    equi_depth_histogram as _equi_depth_histogram,
)

equi_width_histogram = cached(_equi_width_histogram)
equi_depth_histogram = cached(_equi_depth_histogram)
