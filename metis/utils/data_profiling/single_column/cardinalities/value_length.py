from typing import Union
import pandas as pd


def _get_string_lengths(series: pd.Series) -> pd.Series:
    """
    Convert values to strings and calculate their lengths, excluding nulls.
    
    :param series: Input Series.
    :return: Series containing the character lengths of non-null values.
    """
    return series.dropna().astype(str).str.len()


def value_length_min(data: Union[pd.Series, pd.DataFrame]) -> Union[int, pd.Series]:
    """
    Calculate minimum value length in characters.
    
    All values are converted to their string representation and the minimum
    character length is returned. Null values are excluded.
    
    :param data: Input Series (single column) or DataFrame (multiple columns).
    :return: Minimum character length as int if Series input, Series of ints if DataFrame input.
    """
    if isinstance(data, pd.Series):
        lengths = _get_string_lengths(data)
        return int(lengths.min()) if not lengths.empty else 0
    else:
        result = {}
        for col in data.columns:
            lengths = _get_string_lengths(data[col])
            result[col] = int(lengths.min()) if not lengths.empty else 0
        return pd.Series(result)


def value_length_max(data: Union[pd.Series, pd.DataFrame]) -> Union[int, pd.Series]:
    """
    Calculate maximum value length in characters.
    
    All values are converted to their string representation and the maximum
    character length is returned. Null values are excluded.
    
    :param data: Input Series (single column) or DataFrame (multiple columns).
    :return: Maximum character length as int if Series input, Series of ints if DataFrame input.
    """
    if isinstance(data, pd.Series):
        lengths = _get_string_lengths(data)
        return int(lengths.max()) if not lengths.empty else 0
    else:
        result = {}
        for col in data.columns:
            lengths = _get_string_lengths(data[col])
            result[col] = int(lengths.max()) if not lengths.empty else 0
        return pd.Series(result)


def value_length_mean(data: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    """
    Calculate mean value length in characters.
    
    All values are converted to their string representation and the mean
    character length is returned. Null values are excluded.
    
    :param data: Input Series (single column) or DataFrame (multiple columns).
    :return: Mean character length as float if Series input, Series of floats if DataFrame input.
    """
    if isinstance(data, pd.Series):
        lengths = _get_string_lengths(data)
        return float(lengths.mean()) if not lengths.empty else 0.0
    else:
        result = {}
        for col in data.columns:
            lengths = _get_string_lengths(data[col])
            result[col] = float(lengths.mean()) if not lengths.empty else 0.0
        return pd.Series(result)


def value_length_median(data: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    """
    Calculate median value length in characters.
    
    All values are converted to their string representation and the median
    character length is returned. Null values are excluded.
    
    :param data: Input Series (single column) or DataFrame (multiple columns).
    :return: Median character length as float if Series input, Series of floats if DataFrame input.
    """
    if isinstance(data, pd.Series):
        lengths = _get_string_lengths(data)
        return float(lengths.median()) if not lengths.empty else 0.0
    else:
        result = {}
        for col in data.columns:
            lengths = _get_string_lengths(data[col])
            result[col] = float(lengths.median()) if not lengths.empty else 0.0
        return pd.Series(result)