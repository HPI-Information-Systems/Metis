def numeric_similarity(a: float, b: float) -> float:
    """Normalized numeric difference."""
    denom = max(abs(a), abs(b), 1.0)
    return max(0.0, 1.0 - abs(a - b) / denom)