from metis.profiling.cache import cached
from metis.utils.data_profiling.single_column.value_distribution.outliers import (
    detect_outliers as _detect_outliers,
    iqr_bounds as _iqr_bounds,
)

detect_outliers = cached(_detect_outliers)
iqr_bounds = cached(_iqr_bounds)
