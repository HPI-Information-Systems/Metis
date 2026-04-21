from typing import Union
import pandas as pd


def jaccard_similarity(column1: pd.Series, column2: pd.Series) -> float:
    """
    Calculate Jaccard similarity between two columns based on their distinct values.

    Jaccard similarity of columns A and B is defined as:
    |A ∩ B| / |A ∪ B|

    This gives the relative number of distinct values appearing in both columns
    divided by the total number of distinct values in either column. Null values
    are excluded from the calculation.

    :param column1: First input Series.
    :param column2: Second input Series.
    :return: Jaccard similarity score between 0.0 and 1.0, where 1.0 means
             identical sets of distinct values and 0.0 means no overlap.
    """
    set1 = set(column1.dropna().unique())
    set2 = set(column2.dropna().unique())

    if len(set1) == 0 and len(set2) == 0:
        return 1.0

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return float(intersection / union)


def jaccard_similarity_ngrams(column1: pd.Series, column2: pd.Series, n: int = 2) -> float:
    """
    Calculate Jaccard similarity between two columns based on n-gram distributions.

    Since semantically similar values may have different formats, this function
    computes Jaccard similarity of n-gram distributions in the two columns instead
    of exact value matches. This captures similarity at the character level.

    Example: "John" and "Jon" have no exact match, but share an n-gram like "Jo".

    :param column1: First input Series.
    :param column2: Second input Series.
    :param n: Size of n-grams to extract (default: 2 for bigrams).
    :return: Jaccard similarity score between 0.0 and 1.0 based on n-gram overlap.
    """
    def get_ngrams(series: pd.Series, n: int) -> set:
        """Extract all n-grams from a Series."""
        ngrams = set()
        for val in series.dropna().astype(str):
            for i in range(len(val) - n + 1):
                ngrams.add(val[i:i+n])
        return ngrams

    ngrams1 = get_ngrams(column1, n)
    ngrams2 = get_ngrams(column2, n)

    if len(ngrams1) == 0 and len(ngrams2) == 0:
        return 1.0

    intersection = len(ngrams1.intersection(ngrams2))
    union = len(ngrams1.union(ngrams2))

    return float(intersection / union)
