from typing import Union
from decimal import Decimal, InvalidOperation
import pandas as pd


def _to_decimal_string(val: float) -> str:
    """Convert float to full decimal string without scientific notation."""
    try:
        d = Decimal(str(val))
        sign, digits, exponent = d.as_tuple()
        
        if exponent >= 0:
            return ''.join(str(d) for d in digits) + '0' * exponent
        else:
            digits_str = ''.join(str(d) for d in digits)
            if len(digits_str) <= -exponent:
                digits_str = '0' * (-exponent - len(digits_str) + 1) + digits_str
            decimal_pos = len(digits_str) + exponent
            return digits_str[:decimal_pos] + '.' + digits_str[decimal_pos:]
    except (InvalidOperation, ValueError):
        return str(val)


def _calculate_size(series: pd.Series) -> int:
    """
    Calculate the maximum number of digits for numeric values.

    :param series: Input Series.
    :return: Maximum number of digits.
    """
    clean_data = series.dropna()

    if len(clean_data) == 0:
        return 0

    if not pd.api.types.is_numeric_dtype(clean_data):
        try:
            clean_data = pd.to_numeric(clean_data, errors='coerce').dropna()
            if len(clean_data) == 0:
                return 0
        except Exception:
            return 0

    max_digits = 0
    for val in clean_data:
        try:
            val_str = _to_decimal_string(abs(val))
            val_str = val_str.replace('.', '')
            max_digits = max(max_digits, len(val_str))
        except Exception:
            continue

    return max_digits


def _calculate_decimals(series: pd.Series) -> int:
    """
    Calculate the maximum number of decimal places for numeric values.

    :param series: Input Series.
    :return: Maximum number of decimal places.
    """
    clean_data = series.dropna()

    if len(clean_data) == 0:
        return 0

    if not pd.api.types.is_numeric_dtype(clean_data):
        try:
            clean_data = pd.to_numeric(clean_data, errors='coerce').dropna()
            if len(clean_data) == 0:
                return 0
        except Exception:
            return 0

    max_decimals = 0
    for val in clean_data:
        try:
            val_str = _to_decimal_string(val)
            if '.' in val_str:
                decimal_part = val_str.split('.')[1].rstrip('0')
                max_decimals = max(max_decimals, len(decimal_part))
        except Exception:
            continue

    return max_decimals


def size(data: Union[pd.Series, pd.DataFrame]) -> Union[int, pd.Series]:
    """
    Extract the maximum number of digits for decimal, float, and double data types.

    For numeric columns, determines the maximum total number of digits (excluding
    decimal point and sign). Used to determine appropriate data type bounds.
    Returns 0 for non-numeric data.

    :param data: Input Series (single column) or DataFrame (multiple columns).
    :return: Maximum number of digits as int if Series input, Series of ints if
             DataFrame input.
    """
    if isinstance(data, pd.Series):
        return _calculate_size(data)
    else:
        result = {}
        for col in data.columns:
            result[col] = _calculate_size(data[col])
        return pd.Series(result)


def decimals(data: Union[pd.Series, pd.DataFrame]) -> Union[int, pd.Series]:
    """
    Extract the maximum number of decimal places for decimal, float, and double data types.

    For numeric columns with fractional parts, determines the maximum number of
    digits after the decimal point. Used alongside size to determine appropriate
    precision specifications. Returns 0 for integer or non-numeric data.

    :param data: Input Series (single column) or DataFrame (multiple columns).
    :return: Maximum number of decimal places as int if Series input, Series of
             ints if DataFrame input.
    """
    if isinstance(data, pd.Series):
        return _calculate_decimals(data)
    else:
        result = {}
        for col in data.columns:
            result[col] = _calculate_decimals(data[col])
        return pd.Series(result)