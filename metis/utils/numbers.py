def clamp(
    value: int | float, min_value: int | float, max_value: int | float
) -> int | float:
    return max(min(value, max_value), min_value)


def format_count(value: int | float) -> str:
    """Formats a large number with appropriate suffixes (K, M, B, T) for thousands, millions, billions, and trillions."""
    suffixes = ["", "K", "M", "B", "T"]
    string_value = str(int(value))
    suffix = min((len(string_value) - 1) // 3, len(suffixes) - 1)
    return f"{string_value[: -3 * suffix] if suffix > 0 else string_value}{suffixes[suffix]}"
