import pandas as pd
import numpy as np
from metis.utils.similarity_measures.boolean import boolean_similarity
from metis.utils.similarity_measures.category import category_similarity
from metis.utils.similarity_measures.number import numeric_similarity
from metis.utils.similarity_measures.string import normalized_levenshtein_distance
from metis.utils.similarity_measures.timestamp import timestamp_similarity


# Generic row similarity
# -------------------------------
def row_similarity(row_a: pd.Series, row_b: pd.Series) -> float:
    """Compute row-level similarity by column type."""
    sims = []

    for col in row_a.index:
        a, b = row_a[col], row_b[col]

        if pd.isna(a) or pd.isna(b):
            continue

        if isinstance(a, str):
            sims.append(normalized_levenshtein_distance(a, b))
        # numpy's int64 is not a subclass of int, so without np.integer every
        # integer column of a DataFrame would fall through to the 0.0 fallback.
        elif isinstance(a, (int, float, np.integer, np.floating)):
            sims.append(numeric_similarity(a, b))
        elif isinstance(a, bool):
            sims.append(boolean_similarity(a, b))
        elif isinstance(a, pd.Timestamp):
            sims.append(timestamp_similarity(a, b))
        elif pd.api.types.is_categorical_dtype(row_a[col]):
            sims.append(category_similarity(a, b))
        else:
            # Fallback for unknown types
            sims.append(0.0)

    return float(np.mean(sims)) if sims else 0.0