import pandas as pd
from typing import Tuple, List, Dict

from metis.dismis.detection.detectors.detector import DMVDetector
from utils.datetime import datetime_to_numeric

class NanDetector(DMVDetector):
    def __init__(self, target_types: List[str] = ["numeric", "categorical", "date", "text"]):
        self.target_types = target_types
    def __call__(self, dataset: pd.DataFrame, types: Dict[str, str], target_columns: List[str] = None, embeddings: Dict[str, pd.DataFrame] = {}) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
        result = dataset.notnull().astype(int), dataset.notnull().astype(int), {"total": 0}, list(dataset.columns)

        return result

class QuantileDetector(DMVDetector):
    def __init__(self, target_types: List[str] = ["numeric", "categorical", "date", "text"]):
        self.target_types = target_types
    def __call__(self, dataset: pd.DataFrame, types: Dict[str, str], target_columns: List[str] = None, embeddings: Dict[str, pd.DataFrame] = {}) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:

        transformed = pd.DataFrame(index=dataset.index)
        assessed = []

        for column in dataset.columns:
            if types.get(column) == 'numeric':
                transformed[column] = dataset[column].rank(pct=True)
                assessed.append(column)
            elif types.get(column) == 'date':
                transformed[column] = datetime_to_numeric(dataset[column])[0].rank(pct=True)
                assessed.append(column)
            else:
                transformed[column] = 0

        transformed.fillna(-1, inplace=True)
        times = {"total": 0}

        return transformed, (transformed == 0).astype(int), times, assessed


