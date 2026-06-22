from typing import Union, Dict
import pandas as pd


def value_range(data: Union[pd.Series, pd.DataFrame]) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Compute the observed [min, max] of one or more numeric columns.

    Null values are excluded. For non-numeric or empty columns both bounds
    are returned as None.

    :param data: Input Series (single column) or DataFrame (multiple columns).
    :return: ``{"min": float, "max": float}`` for a Series; a dict keyed by
             column name for a DataFrame.
    """
    if isinstance(data, pd.Series):
        clean = data.dropna()
        if len(clean) == 0 or not pd.api.types.is_numeric_dtype(clean):
            return {"min": None, "max": None}
        return {"min": float(clean.min()), "max": float(clean.max())}

    result: Dict[str, Dict[str, float]] = {}
    for col in data.columns:
        clean = data[col].dropna()
        if len(clean) == 0 or not pd.api.types.is_numeric_dtype(clean):
            result[col] = {"min": None, "max": None}
            continue
        result[col] = {"min": float(clean.min()), "max": float(clean.max())}
    return result
