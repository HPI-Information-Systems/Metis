from typing import Dict, List, Tuple

import pandas as pd

from metis.dismis.detection.detectors.detector import DMVDetector
from metis.dismis.utils.datetime import datetime_to_numeric
from metis.dismis.utils.types import COLUMN_TYPES


class NanDetector(DMVDetector):
    def __init__(
        self, target_types: List[str] = ["numeric", "categorical", "date", "text"]
    ):
        self.target_types = target_types

    def __call__(
        self,
        dataset: pd.DataFrame,
        column_types: Dict[str, COLUMN_TYPES],
        target_columns: List[str] | None = None,
        embeddings: Dict[str, pd.DataFrame] = {},
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], List[str]]:
        times: Dict[str, float] = {"total": 0}

        return (
            dataset.notnull().astype(int),
            dataset.notnull().astype(int),
            times,
            list(dataset.columns),
        )


class QuantileDetector(DMVDetector):
    def __init__(
        self, target_types: List[str] = ["numeric", "categorical", "date", "text"]
    ):
        self.target_types = target_types

    def __call__(
        self,
        dataset: pd.DataFrame,
        column_types: Dict[str, COLUMN_TYPES],
        target_columns: List[str] | None = None,
        embeddings: Dict[str, pd.DataFrame] = {},
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], List[str]]:

        transformed = pd.DataFrame(index=dataset.index)
        assessed = []

        for column in dataset.columns:
            if column_types.get(column) == "numeric":
                transformed[column] = dataset[column].rank(pct=True)
                assessed.append(column)
            elif column_types.get(column) == "date":
                transformed[column] = datetime_to_numeric(dataset[column])[0].rank(
                    pct=True
                )
                assessed.append(column)
            else:
                transformed[column] = 0

        transformed.fillna(-1, inplace=True)
        times: Dict[str, float] = {"total": 0}

        return transformed, (transformed == 0).astype(int), times, assessed
