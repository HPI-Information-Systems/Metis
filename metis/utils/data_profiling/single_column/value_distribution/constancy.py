from typing import Any, Union
import pandas as pd


def constancy(data: Union[pd.Series, pd.DataFrame], include_nulls: bool = False) -> Union[float, pd.Series]:
    """
    Calculate constancy as the ratio of the most frequent value's frequency
    to the total number of values.

    Represents the proportion of some constant value compared with the entire
    column. High constancy suggests a column with low variability or a dominant
    default value.

    :param data: Input Series (single column) or DataFrame (multiple columns).
    :param include_nulls: If True, use total row count as denominator (paper definition).
                          If False (default), use non-null count as denominator.
    :return: Constancy ratio (0.0 to 1.0) as float if Series input, Series of
             floats if DataFrame input. Returns 0.0 for empty or all-null data.
    """
    if isinstance(data, pd.Series):
        clean_data = data.dropna()

        if len(clean_data) == 0:
            return 0.0

        max_frequency = clean_data.value_counts().max()
        denominator = len(data) if include_nulls else len(clean_data)

        return float(max_frequency / denominator)
    else:
        result = {}
        for col in data.columns:
            clean_data = data[col].dropna()

            if len(clean_data) == 0:
                result[col] = 0.0
                continue

            max_frequency = clean_data.value_counts().max()
            denominator = len(data) if include_nulls else len(clean_data)

            result[col] = float(max_frequency / denominator)

        return pd.Series(result)


def most_frequent_value(data: Union[pd.Series, pd.DataFrame]) -> Union[Any, pd.Series]:
    """
    Return the most frequent value in the data.

    If multiple values have the same maximum frequency, returns the one that
    comes first in sorted order. Null values are excluded.

    :param data: Input Series (single column) or DataFrame (multiple columns).
    :return: Most frequent value if Series input, Series of most frequent values
             if DataFrame input. Returns None for empty or all-null data.
    """
    if isinstance(data, pd.Series):
        clean_data = data.dropna()

        if len(clean_data) == 0:
            return None

        return clean_data.mode().iloc[0]
    else:
        result = {}
        for col in data.columns:
            clean_data = data[col].dropna()

            if len(clean_data) == 0:
                result[col] = None
                continue

            result[col] = clean_data.mode().iloc[0]

        return pd.Series(result)