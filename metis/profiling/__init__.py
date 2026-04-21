"""Cached data-profiling functions and the ``DataProfileManager``.

Import profiling functions from this package instead of
``metis.utils.data_profiling`` to get automatic caching::

    from metis.profiling import null_count   # cached version

The original (uncached) implementations remain accessible via
``metis.utils.data_profiling``.
"""

from .data_profile_manager import DataProfileManager

# --- Cardinalities ---
from .single_column.cardinalities.distinct_values import distinct_count
from .single_column.cardinalities.null_values import null_count, null_percentage
from .single_column.cardinalities.row_count import row_count
from .single_column.cardinalities.uniqueness import uniqueness
from .single_column.cardinalities.value_length import (
    value_length_max,
    value_length_mean,
    value_length_median,
    value_length_min,
)

# --- Value Distribution ---
from .single_column.value_distribution.histogram import (
    equi_width_histogram,
    equi_depth_histogram,
)
from .single_column.value_distribution.constancy import constancy, most_frequent_value
from .single_column.value_distribution.quartiles import quartiles, interquartile_range

# --- Patterns & Data Types ---
from .single_column.patterns_and_data_types.basic_type import basic_type
from .single_column.patterns_and_data_types.data_type import data_type
from .single_column.patterns_and_data_types.numeric_precision import size, decimals
from .single_column.patterns_and_data_types.patterns import patterns

# --- Summaries & Sketches ---
from .single_column.summaries_and_sketches.jaccard_similarity import (
    jaccard_similarity,
    jaccard_similarity_ngrams,
)
from .single_column.summaries_and_sketches.minhash import (
    minhash_signature,
    estimate_jaccard_from_minhash,
)

# --- Domain Classification ---
from .single_column.domain_classification.data_class import data_class
from .single_column.domain_classification.domain import domain
