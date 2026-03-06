from typing import Union, List, Dict
import pandas as pd
from collections import Counter


def _extract_pattern(value: str) -> str:
    """
    Extract a pattern representation from a string value.

    Pattern codes: A = uppercase letter, a = lowercase letter, 9 = digit,
                   # = special character, space = space, ? = other letter (e.g., CJK).

    :param value: Input string.
    :return: Pattern representation.
    """
    pattern = []
    for char in value:
        if char.isupper():
            pattern.append('A')
        elif char.islower():
            pattern.append('a')
        elif char.isdigit():
            pattern.append('9')
        elif char.isspace():
            pattern.append(' ')
        elif char.isalpha():
            pattern.append('?')
        else:
            pattern.append('#')
    return ''.join(pattern)


def _get_top_patterns(series: pd.Series, top_n: int = 5) -> List[Dict[str, Union[str, int, float]]]:
    """
    Extract the most frequent patterns from a Series.

    :param series: Input Series.
    :param top_n: Number of top patterns to return.
    :return: List of dictionaries containing pattern, count, and frequency.
    """
    clean_data = series.dropna().astype(str)

    if len(clean_data) == 0:
        return []

    pattern_list = [_extract_pattern(val) for val in clean_data]
    pattern_counts = Counter(pattern_list)

    total = len(pattern_list)
    top_patterns = []

    for pattern, count in pattern_counts.most_common(top_n):
        top_patterns.append({
            "pattern": pattern,
            "count": count,
            "frequency": float(count / total)
        })

    return top_patterns


def patterns(
    data: Union[pd.Series, pd.DataFrame], top_n: int = 5
) -> Union[List[Dict[str, Union[str, int, float]]], Dict[str, List[Dict[str, Union[str, int, float]]]]]:
    """
    Extract frequent patterns observed in the data of a column.

    Patterns are expressed using pattern codes where:
    - A = uppercase letter
    - a = lowercase letter
    - 9 = digit
    - # = special character
    - ? = other letter (e.g., CJK characters)
    - (space) = space

    Example: "John123" -> "Aaaa999", "+1 (555) 123-4567" -> "#9 (#99) 999-9999"

    Data that does not conform to discovered patterns is likely erroneous or
    ill-formed. Returns the top N most frequent patterns with their counts
    and frequencies.

    :param data: Input Series (single column) or DataFrame (multiple columns).
    :param top_n: Number of top patterns to return per column (default: 5).
    :return: List of pattern dictionaries if Series input, dict of column names
             to pattern lists if DataFrame input. Each pattern dict contains
             'pattern' (str), 'count' (int), and 'frequency' (float).
    """
    if isinstance(data, pd.Series):
        return _get_top_patterns(data, top_n)
    else:
        result = {}
        for col in data.columns:
            result[col] = _get_top_patterns(data[col], top_n)
        return result