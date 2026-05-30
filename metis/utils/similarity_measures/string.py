# https://stackoverflow.com/a/32558749
def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between s1 and s2."""
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min(distances[i1], distances[i1 + 1], distances_[-1])
                )
        distances = distances_
    return distances[-1]

def normalized_levenshtein_distance(a: str, b: str) -> float:
    """Normalized Levenshtein distance between two strings."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    distance = levenshtein_distance(a.lower(), b.lower())
    return 1.0 - distance / max(len(a), len(b))