from typing import Union
import pandas as pd
from datasketch import MinHash


def minhash_signature(data: Union[pd.Series, pd.DataFrame], num_perm: int = 128) -> Union[MinHash, dict]:
    """
    Create a MinHash signature for efficient Jaccard similarity estimation.

    MinHash creates compact signatures that can be compared efficiently to
    estimate set similarity without computing full set intersections. If distinct
    value sets are not directly available, Jaccard similarity can be estimated
    using MinHash signatures.

    The signature size (num_perm) controls the trade-off between accuracy and
    memory: more permutations give more accurate estimates but use more space.

    :param data: Input Series (single column) or DataFrame (multiple columns).
    :param num_perm: Number of permutations for MinHash (default: 128). Higher
                     values provide more accurate similarity estimates.
    :return: MinHash object if Series input, dict of column names to MinHash
             objects if DataFrame input.
    """
    if isinstance(data, pd.Series):
        m = MinHash(num_perm=num_perm)
        for value in data.dropna().astype(str).unique():
            m.update(value.encode('utf-8'))
        return m
    else:
        result = {}
        for col in data.columns:
            m = MinHash(num_perm=num_perm)
            for value in data[col].dropna().astype(str).unique():
                m.update(value.encode('utf-8'))
            result[col] = m
        return result


def estimate_jaccard_from_minhash(minhash1: MinHash, minhash2: MinHash) -> float:
    """
    Estimate Jaccard similarity between two columns using their MinHash signatures.

    This is much faster than computing exact Jaccard similarity for large datasets,
    while providing a good approximation. The accuracy depends on the num_perm
    parameter used when creating the MinHash signatures.

    :param minhash1: MinHash signature of the first column.
    :param minhash2: MinHash signature of the second column.
    :return: Estimated Jaccard similarity between 0.0 and 1.0.
    """
    return float(minhash1.jaccard(minhash2))
