from typing import Union

import pandas as pd


def null_count(data: Union[pd.Series, pd.DataFrame]) -> Union[int, pd.Series]:
    """
    Count the number of null/missing values.

    :param data: Input Series (single column) or DataFrame (multiple columns).
    :return: Number of null values as int if Series input, Series of ints if DataFrame input.
    """
    result = data.isna().sum()
    return result if isinstance(result, pd.Series) else int(result)


def null_percentage(data: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    """
    Calculate the percentage of null/missing values.

    :param data: Input Series (single column) or DataFrame (multiple columns).
    :return: Percentage of null values (0-100) as float if Series input, Series of floats if DataFrame input.
    """
    if len(data) == 0:
        result = data.isna().sum()
    else:
        result = data.isna().mean() * 100
    return result if isinstance(result, pd.Series) else float(result)
