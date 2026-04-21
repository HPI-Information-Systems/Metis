from typing import Union, Dict, List, Tuple
import pandas as pd
import numpy as np


def equi_width_histogram(
    data: Union[pd.Series, pd.DataFrame], bins: int = 10
) -> Union[Dict[str, Union[List[Tuple[float, float]], List[int]]], Dict[str, Dict[str, Union[List[Tuple[float, float]], List[int]]]]]:
    """
    Create an equi-width histogram where buckets span value ranges of same length.

    Divides the range of values into bins of equal width and counts frequencies
    in each bin. Only works with numeric data. Non-numeric columns are skipped.

    :param data: Input Series (single column) or DataFrame (multiple columns).
    :param bins: Number of bins to create (default: 10).
    :return: Dictionary containing 'bin_edges' (list of tuples with min/max of each bin)
             and 'frequencies' (counts per bin). For DataFrame input, returns dict of
             column names to their histogram dicts.
    """
    if isinstance(data, pd.Series):
        clean_data = data.dropna()

        if len(clean_data) == 0 or not pd.api.types.is_numeric_dtype(clean_data):
            return {"bin_edges": [], "frequencies": []}

        counts, bin_edges = np.histogram(clean_data, bins=bins)
        bin_ranges = [(float(bin_edges[i]), float(bin_edges[i + 1])) for i in range(len(bin_edges) - 1)]

        return {
            "bin_edges": bin_ranges,
            "frequencies": [int(c) for c in counts]
        }
    else:
        result = {}
        for col in data.columns:
            clean_data = data[col].dropna()

            if len(clean_data) == 0 or not pd.api.types.is_numeric_dtype(clean_data):
                result[col] = {"bin_edges": [], "frequencies": []}
                continue

            counts, bin_edges = np.histogram(clean_data, bins=bins)
            bin_ranges = [(float(bin_edges[i]), float(bin_edges[i + 1])) for i in range(len(bin_edges) - 1)]

            result[col] = {
                "bin_edges": bin_ranges,
                "frequencies": [int(c) for c in counts]
            }

        return result


def equi_depth_histogram(
    data: Union[pd.Series, pd.DataFrame], bins: int = 10
) -> Union[Dict[str, Union[List[Tuple[float, float]], List[int]]], Dict[str, Dict[str, Union[List[Tuple[float, float]], List[int]]]]]:
    """
    Create an equi-depth (equi-height) histogram where each bucket represents
    approximately the same number of value occurrences.

    Uses quantiles to determine bin boundaries such that each bin contains roughly
    the same number of values. Only works with numeric data. Non-numeric columns are skipped.

    :param data: Input Series (single column) or DataFrame (multiple columns).
    :param bins: Number of bins to create (default: 10).
    :return: Dictionary containing 'bin_edges' (list of tuples with min/max of each bin)
             and 'frequencies' (counts per bin). For DataFrame input, returns dict of
             column names to their histogram dicts.
    """
    if isinstance(data, pd.Series):
        clean_data = data.dropna()

        if len(clean_data) == 0 or not pd.api.types.is_numeric_dtype(clean_data):
            return {"bin_edges": [], "frequencies": []}

        quantiles = np.linspace(0, 1, bins + 1)
        bin_edges = clean_data.quantile(quantiles).values
        bin_edges = np.unique(bin_edges)

        if len(bin_edges) <= 1:
            return {"bin_edges": [], "frequencies": []}

        counts, _ = np.histogram(clean_data, bins=bin_edges)
        bin_ranges = [(float(bin_edges[i]), float(bin_edges[i + 1])) for i in range(len(bin_edges) - 1)]

        return {
            "bin_edges": bin_ranges,
            "frequencies": [int(c) for c in counts]
        }
    else:
        result = {}
        for col in data.columns:
            clean_data = data[col].dropna()

            if len(clean_data) == 0 or not pd.api.types.is_numeric_dtype(clean_data):
                result[col] = {"bin_edges": [], "frequencies": []}
                continue

            quantiles = np.linspace(0, 1, bins + 1)
            bin_edges = clean_data.quantile(quantiles).values
            bin_edges = np.unique(bin_edges)

            if len(bin_edges) <= 1:
                result[col] = {"bin_edges": [], "frequencies": []}
                continue

            counts, _ = np.histogram(clean_data, bins=bin_edges)
            bin_ranges = [(float(bin_edges[i]), float(bin_edges[i + 1])) for i in range(len(bin_edges) - 1)]

            result[col] = {
                "bin_edges": bin_ranges,
                "frequencies": [int(c) for c in counts]
            }

        return result

