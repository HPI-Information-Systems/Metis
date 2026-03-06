from metis.profiling.cache import cached
from metis.utils.data_profiling.single_column.domain_classification.domain import (
    domain as _domain,
)

domain = cached(_domain)
