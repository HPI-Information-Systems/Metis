from typing import Union
import pandas as pd


def distinct_count(data: Union[pd.Series, pd.DataFrame]) -> Union[int, pd.Series]:
    """
    Count the number of distinct (unique) values, excluding nulls.
    
    :param data: Input Series (single column) or DataFrame (multiple columns).
    :return: Number of distinct values as int if Series input, Series of ints if DataFrame input.
    """
    result = data.nunique()
    return int(result) if isinstance(data, pd.Series) else result