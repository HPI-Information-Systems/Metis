import pandas as pd

def timestamp_similarity(a: pd.Timestamp, b: pd.Timestamp) -> float:
    """Returns similarity based on relative difference in seconds."""
    delta = abs((a - b).total_seconds())
    max_delta = max(abs(a.timestamp()), abs(b.timestamp()), 1.0)
    return max(0.0, 1.0 - delta / max_delta)