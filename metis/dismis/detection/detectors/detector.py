from typing import Dict, List, Tuple

import pandas as pd

from metis.dismis.utils.types import COLUMN_TYPES


class DMVDetector:
    def __call__(
        self,
        dataset: pd.DataFrame,
        column_types: Dict[str, COLUMN_TYPES],
        target_columns: List[str] | None = None,
        embeddings: Dict[str, pd.DataFrame] = {},
    ) -> (
        Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], List[str]]
        | Dict[str, Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], List[str]]]
    ):
        """
        Detects disguised missing values (DMVs) in the dataset.
        Args:
            dataset (pd.DataFrame): The dataset to analyze for DMVs.
            target_columns (list): List of columns to check for DMVs.
        Returns:
            pd.DataFrame: A DataFrame with the same structure as the input, but with detection scores between 0 and 1.
            pd.DataFrame: A DataFrame with the same structure as the input, but with binary detection labels.
            Dict[str, float]: A Dictionary containing the runtimes of individual detection steps.
            List[str]: A List of columns that were analyzed.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
