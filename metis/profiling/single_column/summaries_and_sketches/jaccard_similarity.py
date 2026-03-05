from metis.profiling.cache import cached
from metis.utils.data_profiling.single_column.summaries_and_sketches.jaccard_similarity import (
    jaccard_similarity as _jaccard_similarity,
    jaccard_similarity_ngrams as _jaccard_similarity_ngrams,
)

jaccard_similarity = cached(_jaccard_similarity)
jaccard_similarity_ngrams = cached(_jaccard_similarity_ngrams)
