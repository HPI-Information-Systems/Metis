from typing import Union
import pandas as pd


def _infer_sql_type(series: pd.Series) -> str:
    """
    Infer the most specific SQL data type for a Series.

    :param series: Input Series.
    :return: SQL type name as string.
    """
    clean_data = series.dropna()

    if len(clean_data) == 0:
        return "varchar"

    if pd.api.types.is_bool_dtype(clean_data):
        return "boolean"

    if pd.api.types.is_datetime64_any_dtype(clean_data):
        return "timestamp"

    if pd.api.types.is_integer_dtype(clean_data):
        max_val = clean_data.abs().max()
        if max_val <= 32767:
            return "smallint"
        elif max_val <= 2147483647:
            return "int"
        elif max_val <= 9223372036854775807:
            return "bigint"
        else:
            return "numeric"

    if pd.api.types.is_float_dtype(clean_data):
        return "double"

    sample_size = min(100, len(clean_data))
    sample = clean_data.sample(n=sample_size, random_state=42).astype(str)

    date_pattern = r'^\d{4}-\d{2}-\d{2}$'
    if sample.str.match(date_pattern, na=False).mean() > 0.8:
        return "date"

    time_pattern = r'^\d{2}:\d{2}:\d{2}$'
    if sample.str.match(time_pattern, na=False).mean() > 0.8:
        return "time"

    timestamp_pattern = r'^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}$'
    if sample.str.match(timestamp_pattern, na=False).mean() > 0.8:
        return "timestamp"

    try:
        numeric_converted = pd.to_numeric(clean_data, errors='coerce')
        valid_numeric = numeric_converted.dropna()
        if len(valid_numeric) > 0 and numeric_converted.notna().mean() > 0.8:
            if (valid_numeric % 1 == 0).all():
                return "int"
            else:
                return "double"
    except Exception:
        pass

    max_length = clean_data.astype(str).str.len().max()
    if max_length <= 255:
        return "varchar"
    else:
        return "text"

def data_type(data: Union[pd.Series, pd.DataFrame]) -> Union[str, pd.Series]:
    """
    Infer the concrete DBMS-specific data type (SQL types).

    Data of many types must follow fixed, sometimes DBMS-specific patterns.
    When classifying, chooses the most specific data type possible, avoiding
    catchalls like char or varchar if possible.

    Possible return values: 'boolean', 'smallint', 'int', 'bigint', 'numeric',
    'double', 'date', 'time', 'timestamp', 'varchar', 'text'.

    :param data: Input Series (single column) or DataFrame (multiple columns).
    :return: SQL type as string if Series input, Series of type strings if
             DataFrame input.
    """
    if isinstance(data, pd.Series):
        return _infer_sql_type(data)
    else:
        result = {}
        for col in data.columns:
            result[col] = _infer_sql_type(data[col])
        return pd.Series(result)
