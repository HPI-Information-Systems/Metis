from typing import Union
import pandas as pd


def _infer_basic_type(series: pd.Series) -> str:
    """
    Infer the basic type of a Series by examining its values.

    :param series: Input Series.
    :return: One of: 'numeric', 'alphabetic', 'alphanumeric', 'date', 'time', 'mixed', 'empty'.
    """
    clean_data = series.dropna()

    if len(clean_data) == 0:
        return "empty"

    if pd.api.types.is_numeric_dtype(clean_data):
        return "numeric"

    if pd.api.types.is_datetime64_any_dtype(clean_data):
        return "date"

    sample_size = min(100, len(clean_data))
    sample = clean_data.sample(n=sample_size, random_state=42).astype(str)

    datetime_pattern = r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}([T\s]\d{1,2}:\d{2}(:\d{2})?)?$'
    is_datetime = sample.str.match(datetime_pattern, na=False).mean() > 0.8

    time_pattern = r'^\d{1,2}:\d{2}(:\d{2})?(\s?[AaPp][Mm])?$'
    is_time = sample.str.match(time_pattern, na=False).mean() > 0.8

    if is_datetime:
        return "date"
    if is_time:
        return "time"

    # Scientific notation pattern
    scientific_pattern = r'^-?\d+\.?\d*[eE][+-]?\d+$'
    is_scientific = sample.str.match(scientific_pattern, na=False).mean() > 0.8

    # General numeric string pattern (integers, decimals, negatives)
    numeric_string_pattern = r'^-?\d+\.?\d*$'
    is_numeric_string = sample.str.match(numeric_string_pattern, na=False).mean() > 0.8

    if is_scientific or is_numeric_string:
        return "numeric"

    has_digit = sample.str.contains(r'\d', regex=True, na=False).any()
    has_alpha = sample.str.contains(r'[a-zA-Z]', regex=True, na=False).any()

    if has_digit and has_alpha:
        return "alphanumeric"
    elif has_alpha and not has_digit:
        return "alphabetic"
    else:
        return "mixed"


def basic_type(data: Union[pd.Series, pd.DataFrame]) -> Union[str, pd.Series]:
    """
    Classify a column as numeric, alphabetic, alphanumeric, date, or time.

    Determined by examining the presence or absence of numeric and non-numeric
    characters. Date and time are recognized by numbers within certain ranges
    and numbers separated in regular patterns by special symbols.

    :param data: Input Series (single column) or DataFrame (multiple columns).
    :return: Basic type as string if Series input ('numeric', 'alphabetic',
             'alphanumeric', 'date', 'time', 'mixed', 'empty'), Series of type
             strings if DataFrame input.
    """
    if isinstance(data, pd.Series):
        return _infer_basic_type(data)
    else:
        result = {}
        for col in data.columns:
            result[col] = _infer_basic_type(data[col])
        return pd.Series(result)
