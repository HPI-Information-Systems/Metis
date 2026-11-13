"""Errors and checks for datasets that cannot be assessed.

A dataset that is missing, unreadable or empty is rejected before any metric
runs. The check lives here and is applied by the orchestrator and the GUI, so
every metric declines such a dataset the same way and a metric author does
not have to handle it.
"""
from __future__ import annotations

import pandas as pd


class DatasetError(ValueError):
    """Base class for a dataset that cannot be assessed."""


class DatasetLoadError(DatasetError):
    """Raised when a dataset file is missing or cannot be read as a table."""


class EmptyDatasetError(DatasetError):
    """Raised when a dataset has no rows or no columns."""


def ensure_not_empty(data: pd.DataFrame, name: str = "The dataset") -> None:
    """
    Reject a dataset that has nothing to assess.

    :param data: The dataset about to be assessed.
    :param name: How to refer to the dataset in the error message.
    :raises EmptyDatasetError: If the dataset has no columns or no rows.
    :return: None.
    """
    n_rows, n_cols = data.shape
    if n_cols == 0:
        raise EmptyDatasetError(
            f"{name} is empty: it has no columns. There is nothing to assess."
        )
    if n_rows == 0:
        raise EmptyDatasetError(
            f"{name} is empty: it has {n_cols} column(s) but no rows. "
            "There is nothing to assess."
        )
