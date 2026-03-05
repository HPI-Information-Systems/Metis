from typing import Union
import pandas as pd


def uniqueness(data: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    """
    Calculate uniqueness as the ratio of distinct values to total rows.
    Uniqueness = distinct_count / row_count
    
    Null values are excluded from the distinct count, meaning a column with all
    null values will have uniqueness of 0.0.

    :param data: Input Series (single column) or DataFrame (multiple columns).
    :return: Uniqueness ratio (0.0 to 1.0) as float if Series input, Series of floats if DataFrame input.
    """
    if len(data) == 0:
        return 0.0 if isinstance(data, pd.Series) else pd.Series(dtype=float)

    if isinstance(data, pd.Series):
        return float(data.nunique() / len(data))
    else:
        return data.nunique() / len(data)
