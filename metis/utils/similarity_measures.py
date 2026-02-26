import pandas as pd
import numpy as np
import Levenshtein

# -------------------------------
# String similarity
# -------------------------------
def levenshtein_similarity(a: str, b: str) -> float:
    """Normalized Levenshtein similarity between two strings."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    distance = Levenshtein.distance(a.lower(), b.lower())
    return 1.0 - distance / max(len(a), len(b))

# -------------------------------
# Numeric similarity
# -------------------------------
def numeric_similarity(a: float, b: float) -> float:
    """Normalized numeric difference."""
    denom = max(abs(a), abs(b), 1.0)
    return max(0.0, 1.0 - abs(a - b) / denom)

# -------------------------------
# Timestamp similarity
# -------------------------------
def timestamp_similarity(a: pd.Timestamp, b: pd.Timestamp) -> float:
    """Returns similarity based on relative difference in seconds."""
    delta = abs((a - b).total_seconds())
    max_delta = max(abs(a.timestamp()), abs(b.timestamp()), 1.0)
    return max(0.0, 1.0 - delta / max_delta)

# -------------------------------
# Category similarity
# -------------------------------
def category_similarity(a: any, b: any) -> float:
    return 1.0 if a == b else 0.0

# -------------------------------
# Boolean similarity
# -------------------------------
def boolean_similarity(a: bool, b: bool) -> float:
    return 1.0 if a == b else 0.0

# -------------------------------
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
            sims.append(levenshtein_similarity(a, b))
        elif isinstance(a, (int, float)):
            sims.append(numeric_similarity(a, b))
        elif isinstance(a, bool):
            sims.append(boolean_similarity(a, b))
        elif isinstance(a, pd.Timestamp):
            sims.append(timestamp_similarity(a, b))
        elif pd.api.types.is_categorical_dtype(row_a[col]):
            sims.append(category_similarity(a, b))
        else:
            # Fallback für unbekannte Typen
            sims.append(0.0)

    return float(np.mean(sims)) if sims else 0.0