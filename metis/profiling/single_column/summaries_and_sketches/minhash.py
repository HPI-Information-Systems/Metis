from metis.profiling.cache import cached
from metis.utils.data_profiling.single_column.summaries_and_sketches.minhash import (
    minhash_signature as _minhash_signature,
    estimate_jaccard_from_minhash as _estimate_jaccard_from_minhash,
)

minhash_signature = cached(_minhash_signature)
estimate_jaccard_from_minhash = _estimate_jaccard_from_minhash
