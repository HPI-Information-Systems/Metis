from typing import Union
import pandas as pd


def _classify_data_class(series: pd.Series) -> str:
    """
    Classify the semantic data class of a Series.

    :param series: Input Series.
    :return: Data class as string.
    """
    clean_data = series.dropna()

    if len(clean_data) == 0:
        return "unknown"

    if pd.api.types.is_datetime64_any_dtype(clean_data):
        return "date/time"

    if pd.api.types.is_bool_dtype(clean_data):
        return "indicator"

    if pd.api.types.is_numeric_dtype(clean_data):
        distinct_count = clean_data.nunique()
        total_count = len(clean_data)

        if distinct_count == total_count or distinct_count / total_count > 0.95:
            return "identifier"
        else:
            return "quantity"

    sample_size = min(100, len(clean_data))
    sample = clean_data.sample(n=sample_size, random_state=42).astype(str)

    avg_length = sample.str.len().mean()
    distinct_ratio = clean_data.nunique() / len(clean_data)

    if distinct_ratio > 0.95:
        return "identifier"

    if distinct_ratio < 0.05:
        return "code"

    code_pattern = r'^[A-Za-z0-9]{2,10}$'
    if avg_length <= 10 and sample.str.match(code_pattern, na=False).mean() > 0.7:
        return "code"

    if avg_length > 50:
        return "text"

    return "text"


def data_class(data: Union[pd.Series, pd.DataFrame]) -> Union[str, pd.Series]:
    """
    Classify the semantic, generic data type.

    Goes beyond syntactic patterns to categorize the semantic role of the column.
    Possible classifications:
    - code: Short alphanumeric codes with low cardinality (e.g., country codes, status codes)
    - indicator: Boolean or binary values
    - text: Free-form text with variable length
    - date/time: Temporal data
    - quantity: Numeric measurements or amounts
    - identifier: High-cardinality unique or near-unique values (e.g., IDs, keys)

    :param data: Input Series (single column) or DataFrame (multiple columns).
    :return: Data class as string if Series input, Series of data class strings
             if DataFrame input.
    """
    if isinstance(data, pd.Series):
        return _classify_data_class(data)
    else:
        result = {}
        for col in data.columns:
            result[col] = _classify_data_class(data[col])
        return pd.Series(result)