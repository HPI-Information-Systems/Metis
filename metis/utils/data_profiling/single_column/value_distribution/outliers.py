from typing import Callable, Dict, Union
import pandas as pd

# Strategy signature: (series_with_nans_kept_as_NaN, **params) -> bool Series
OutlierFn = Callable[..., pd.Series]
_REGISTRY: Dict[str, OutlierFn] = {}


def register_outlier_method(name: str, fn: OutlierFn) -> None:
    """Register an outlier-detection strategy under ``name``.

    Strategies receive a Series (NaN values preserved so positional alignment
    of the returned mask matches the input) plus keyword params. They must
    return a boolean Series of the same length where True marks an outlier
    and False marks an inlier. NaN positions should be returned as False.
    """
    _REGISTRY[name] = fn


def available_methods() -> list[str]:
    return sorted(_REGISTRY)


def iqr_bounds(series: pd.Series, multiplier: float = 1.5) -> Dict[str, float]:
    """Return the Tukey IQR bounds and intermediate stats for ``series``.

    Returns NaN-filled bounds for non-numeric or all-NaN columns so the
    metric layer can still record an explanation row.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return {"Q1": None, "Q3": None, "IQR": None,
                "lower_bound": None, "upper_bound": None,
                "multiplier": float(multiplier)}
    clean = series.dropna()
    if clean.empty:
        return {"Q1": None, "Q3": None, "IQR": None,
                "lower_bound": None, "upper_bound": None,
                "multiplier": float(multiplier)}
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    return {
        "Q1": q1, "Q3": q3, "IQR": iqr,
        "lower_bound": q1 - multiplier * iqr,
        "upper_bound": q3 + multiplier * iqr,
        "multiplier": float(multiplier),
    }


def _iqr_outliers(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Tukey / IQR outlier detection. Non-numeric or all-NaN → all False."""
    bounds = iqr_bounds(series, multiplier=multiplier)
    if bounds["lower_bound"] is None:
        return pd.Series(False, index=series.index)
    mask = (series < bounds["lower_bound"]) | (series > bounds["upper_bound"])
    return mask.fillna(False)


register_outlier_method("iqr", _iqr_outliers)


def detect_outliers(
    data: Union[pd.Series, pd.DataFrame],
    method: str = "iqr",
    **params,
) -> Union[pd.Series, Dict[str, pd.Series]]:
    """Apply the named outlier-detection method to one or more columns."""
    if method not in _REGISTRY:
        raise ValueError(
            f"Unknown outlier method '{method}'. Available: {available_methods()}"
        )
    fn = _REGISTRY[method]

    if isinstance(data, pd.Series):
        return fn(data, **params)

    return {col: fn(data[col], **params) for col in data.columns}
