from typing import Union, Dict
import pandas as pd


def quartiles(data: Union[pd.Series, pd.DataFrame]) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Calculate quartiles (Q1, Q2/median, Q3) that divide numeric values into
    four equal groups.

    A special case of equi-depth histogram with exactly four buckets. Only works
    with numeric data. For non-numeric columns, returns None values. Null values
    are excluded from the calculation.

    :param data: Input Series (single column) or DataFrame (multiple columns).
    :return: Dictionary with keys 'Q1', 'Q2', 'Q3' containing the quartile values
             as floats. For DataFrame input, returns dict of column names to their
             quartile dicts. Returns None for quartiles if data is non-numeric or empty.
    """
    if isinstance(data, pd.Series):
        clean_data = data.dropna()

        if len(clean_data) == 0 or not pd.api.types.is_numeric_dtype(clean_data):
            return {"Q1": None, "Q2": None, "Q3": None}

        q1 = float(clean_data.quantile(0.25))
        q2 = float(clean_data.quantile(0.50))
        q3 = float(clean_data.quantile(0.75))

        return {"Q1": q1, "Q2": q2, "Q3": q3}
    else:
        result = {}
        for col in data.columns:
            clean_data = data[col].dropna()

            if len(clean_data) == 0 or not pd.api.types.is_numeric_dtype(clean_data):
                result[col] = {"Q1": None, "Q2": None, "Q3": None}
                continue

            q1 = float(clean_data.quantile(0.25))
            q2 = float(clean_data.quantile(0.50))
            q3 = float(clean_data.quantile(0.75))

            result[col] = {"Q1": q1, "Q2": q2, "Q3": q3}

        return result


def interquartile_range(data: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    """
    Calculate the interquartile range (IQR = Q3 - Q1).

    The IQR represents the range of the middle 50% of the data and is useful
    for detecting outliers. Only works with numeric data. Null values are excluded.

    :param data: Input Series (single column) or DataFrame (multiple columns).
    :return: IQR as float if Series input, Series of floats if DataFrame input.
             Returns None for non-numeric or empty data.
    """
    if isinstance(data, pd.Series):
        clean_data = data.dropna()

        if len(clean_data) == 0 or not pd.api.types.is_numeric_dtype(clean_data):
            return None

        q1 = clean_data.quantile(0.25)
        q3 = clean_data.quantile(0.75)

        return float(q3 - q1)
    else:
        result = {}
        for col in data.columns:
            clean_data = data[col].dropna()

            if len(clean_data) == 0 or not pd.api.types.is_numeric_dtype(clean_data):
                result[col] = None
                continue

            q1 = clean_data.quantile(0.25)
            q3 = clean_data.quantile(0.75)

            result[col] = float(q3 - q1)

        return pd.Series(result)
