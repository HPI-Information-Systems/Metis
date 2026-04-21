from typing import Union
import pandas as pd


def row_count(data: Union[pd.Series, pd.DataFrame]) -> Union[int, pd.Series]:
    """
    Count the total number of rows (including null values).

    :param data: Input Series (single column) or DataFrame (multiple columns).
    :return: Total number of rows as int for Series input, Series of ints for DataFrame input.
    """
    if isinstance(data, pd.Series):
        return int(len(data))
    else:
        # Each column has the same row count (length of the dataframe)
        return pd.Series({col: len(data) for col in data.columns})
