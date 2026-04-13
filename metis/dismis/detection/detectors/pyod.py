import hashlib
import time
from collections import OrderedDict
from functools import wraps
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.gmm import GMM
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.mad import MAD
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from metis.dismis.detection.detectors.detector import DMVDetector
from metis.dismis.detection.detectors.distribution import DistributionFitDetector
from metis.dismis.detection.detectors.utils import (
    split_datetime_column,
    split_mixed_column,
)
from metis.dismis.utils.datetime import datetime_to_numeric
from metis.dismis.utils.types import COLUMN_TYPES


def series_hash(s: pd.Series) -> str:
    # robust hash of values + index
    return hashlib.sha1(
        np.array(pd.util.hash_pandas_object(s, index=True).values)
    ).hexdigest()


def cache_with_limit(maxsize=128):
    def decorator(func):
        cache = OrderedDict()

        @wraps(func)
        def wrapper(series, *args, **kwargs):
            key = (series_hash(series), args, frozenset(kwargs.items()))
            if key in cache:
                # move to end to mark as recently used
                cache.move_to_end(key)
                # print("cache hit")
                return cache[key]
            # compute result
            result = func(series, *args, **kwargs)
            cache[key] = result
            # evict oldest if over limit
            if len(cache) > maxsize:
                cache.popitem(last=False)
            return result

        return wrapper

    return decorator


class ScikitPosthocsESDDetector:
    def __init__(self, max_outliers=10, alpha=0.05):
        self.max_outliers = max_outliers
        self.alpha = alpha

    def fit(self, X):
        X = np.asarray(X).flatten()
        result = sp.outliers_gesd(
            X, alpha=self.alpha, hypo=False, outliers=self.max_outliers
        )

        # result is a DataFrame with 'value', 'R', 'lambda', 'outlier'
        self.labels_ = result["outlier"].astype(int)
        self.decision_scores_ = result["R"]


class RobustZScoreDetector:
    def fit(self, X):
        self.median = np.median(X)
        self.mad = np.median(np.abs(X - self.median))
        self.decision_scores_ = np.abs(X - self.median) / (self.mad + 1e-6)
        self.labels_ = (self.decision_scores_ > 3).astype(int)  # threshold can be tuned


class QuantileDetector:
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X):
        self.q_low = np.quantile(X, self.lower)
        self.q_high = np.quantile(X, self.upper)
        self.decision_scores_ = np.zeros_like(X)
        self.labels_ = ((X <= self.q_low) | (X >= self.q_high)).astype(int)


class PyODDetector(DMVDetector):
    def __init__(
        self,
        detector_name: str,
        target_types: List[str] = ["numeric", "categorical", "text"],
        **kwargs,
    ):
        """
        Initialize the PYODDetector with a specific detector.

        Args:
            detector_name (str): The name of the detector to use.
            **kwargs: Additional keyword arguments to pass to the detector's constructor.
        """
        self.detector_name = detector_name
        self.detector = self._get_detector(detector_name)(**kwargs)
        self.target_types = target_types

    def _get_detector(self, name: str):
        """
        Get the appropriate detector based on the name.

        Args:
            name (str): The name of the detector.

        Returns:
            A PyOD detector instance.
        """

        detectors = {
            "KNN": KNN,
            "LOF": LOF,
            "IForest": IForest,
            "CBLOF": CBLOF,
            "LODA": LODA,
            "HBOS": HBOS,
            "LOF": LOF,
            "COF": COF,
            "GMM": GMM,
            "RobustZ": RobustZScoreDetector,
            "Quantile": QuantileDetector,
            "ESD": ScikitPosthocsESDDetector,
            "MAD": MAD,
        }

        return detectors[name]

    def _extract_features(
        self, column: pd.Series, type: str, embeddings: Dict[str, pd.DataFrame]
    ) -> np.ndarray:
        """
        Extract features from a pandas Series for the detector.

        Args:
            column (pd.Series): The column to extract features from.

        Returns:
            np.ndarray: The extracted features.
        """
        scaler = MinMaxScaler()
        if type == "categorical":
            encoder = OneHotEncoder()
            encoded = encoder.fit_transform(column.to_numpy().reshape(-1, 1))
            return encoded
        elif type in ["numeric", "date"]:
            if type == "date":
                # numeric_features, _, _ = datetime_to_numeric(column) #split_datetime_column(column, column.name)
                numeric_features = split_datetime_column(column, str(column.name))
            else:
                # numeric_features = pd.to_numeric(column, errors='coerce').mask(np.isinf) #split_mixed_column(column, column.name)
                numeric_features = split_mixed_column(column, str(column.name))
            if numeric_features[f"{column.name}_null"].sum() == 0:
                numeric_features.drop(columns=[f"{column.name}_null"], inplace=True)
            if numeric_features[f"{column.name}_str"].sum() == 0:
                numeric_features.drop(columns=[f"{column.name}_str"], inplace=True)
            print(
                "Min - Max for ",
                column.name,
                numeric_features[f"{column.name}_num"].min(),
                numeric_features[f"{column.name}_num"].max(),
            )
            numeric_features[f"{column.name}_num"] = scaler.fit_transform(
                numeric_features[[f"{column.name}_num"]]
            )
            return numeric_features.values
            # numeric_features = scaler.fit_transform(numeric_features.values.reshape(-1, 1))
            # return numeric_features#.values
        elif type == "text":
            return embeddings[str(column.name)].to_numpy()
        else:
            raise ValueError(f"Unsupported type for column {column.name}: {type}")

    def __call__(
        self,
        dataset: pd.DataFrame,
        column_types: Dict[str, COLUMN_TYPES],
        target_columns: List[str] | None = None,
        embeddings: Dict[str, pd.DataFrame] = {},
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], List[str]]:

        times: Dict[str, float] = {
            "extraction": 0,
            "fitting": 0,
            "scoring": 0,
        }
        total_starttime = time.time()
        assessed = []

        df_detect = dataset.copy()
        df_score = pd.DataFrame(0.0, index=dataset.index, columns=dataset.columns)
        df_predict = pd.DataFrame(0, index=dataset.index, columns=dataset.columns)

        columns = dataset.columns if target_columns is None else target_columns

        for column in columns:
            if column_types[column] not in self.target_types:
                continue
            extraction_starttime = time.time()
            target = (
                df_detect[column].dropna()
                if self.detector_name == "MAD"
                else df_detect[column]
            )
            if len(target.dropna()) == 0:
                continue
            target_values = self._extract_features(
                target, column_types[column], embeddings
            )

            if self.detector_name == "MAD":
                target_values = (
                    target_values[:, 0].reshape(-1, 1)
                    if target_values.ndim > 1
                    else target_values.reshape(-1, 1)
                )
            times["extraction"] += time.time() - extraction_starttime

            target_idx = target.index

            fitting_starttime = time.time()
            # print(column, target_values.shape)
            # print(target_values)
            self.detector.fit(target_values)
            times["fitting"] += time.time() - fitting_starttime

            scoring_starttime = time.time()
            scores = np.minimum(self.detector.decision_scores_, 10000)
            # scores = self.detector.predict_proba(target_values)[:, 1]
            # scores[np.isnan(target_values[:, 0])] = 1.0  # assign 1 score to rows with NaN values
            if not np.all(scores == scores[0]):
                df_score.loc[target_idx, column] = scores

            predictions = self.detector.labels_
            assessed.append(column)
            df_predict.loc[target_idx, column] = predictions
            times["scoring"] += time.time() - scoring_starttime

        times["total"] = time.time() - total_starttime

        return df_score, df_predict.astype(int), times, assessed


class PyODDetector2(DMVDetector):
    def __init__(
        self,
        detector_name: str,
        target_types: List[str] = ["numeric", "categorical", "text"],
        nanvalue: float = 0,
        **kwargs,
    ):
        """
        Initialize the PYODDetector with a specific detector.

        Args:
            detector_name (str): The name of the detector to use.
            **kwargs: Additional keyword arguments to pass to the detector's constructor.
        """
        self.detector_name = detector_name
        self.detector = self._get_detector(detector_name)(**kwargs)
        self.target_types = target_types
        self.nanvalue = nanvalue

    def _get_detector(self, name: str):
        """
        Get the appropriate detector based on the name.

        Args:
            name (str): The name of the detector.

        Returns:
            A PyOD detector instance.
        """

        detectors = {
            "KNN": KNN,
            "LOF": LOF,
            "IForest": IForest,
            "CBLOF": CBLOF,
            "LODA": LODA,
            "HBOS": HBOS,
            "LOF": LOF,
            "COF": COF,
            "GMM": GMM,
            "RobustZ": RobustZScoreDetector,
            "Quantile": QuantileDetector,
            "ESD": ScikitPosthocsESDDetector,
            "MAD": MAD,
        }

        return detectors[name]

    def _extract_features(
        self, column: pd.Series, type: str, embeddings: Dict[str, pd.DataFrame]
    ) -> np.ndarray:
        """
        Extract features from a pandas Series for the detector.

        Args:
            column (pd.Series): The column to extract features from.

        Returns:
            np.ndarray: The extracted features.
        """
        scaler = MinMaxScaler()
        if type == "categorical":
            encoder = OneHotEncoder()
            encoded = encoder.fit_transform(column.to_numpy().reshape(-1, 1))
            return encoded
        elif type in ["numeric", "date"]:
            if type == "date":
                numeric_features, _, _ = datetime_to_numeric(
                    column
                )  # split_datetime_column(column, column.name)
                # numeric_features = split_datetime_column(column, column.name)
            else:
                numeric_features = pd.to_numeric(column, errors="coerce").mask(
                    np.isinf
                )  # split_mixed_column(column, column.name)
            #     numeric_features = split_mixed_column(column, column.name)
            # if numeric_features[f"{column.name}_null"].sum() == 0:
            #     numeric_features.drop(columns=[f"{column.name}_null"], inplace=True)
            # if numeric_features[f"{column.name}_str"].sum() == 0:
            #     numeric_features.drop(columns=[f"{column.name}_str"], inplace=True)
            # print("Min - Max for ", column.name, numeric_features[f"{column.name}_num"].min(), numeric_features[f"{column.name}_num"].max())
            # numeric_features[f"{column.name}_num"] = scaler.fit_transform(numeric_features[[f"{column.name}_num"]])
            # return numeric_features.values
            numeric_features = scaler.fit_transform(
                numeric_features.to_numpy().reshape(-1, 1)
            )
            return numeric_features  # .values
        elif type == "text":
            return embeddings[str(column.name)].to_numpy()
        else:
            raise ValueError(f"Unsupported type for column {column.name}: {type}")

    def __call__(
        self,
        dataset: pd.DataFrame,
        column_types: Dict[str, COLUMN_TYPES],
        target_columns: List[str] | None = None,
        embeddings: Dict[str, pd.DataFrame] = {},
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], List[str]]:

        times: Dict[str, float] = {
            "extraction": 0,
            "fitting": 0,
            "scoring": 0,
        }
        total_starttime = time.time()
        assessed = []

        df_detect = dataset.copy()
        df_score = pd.DataFrame(0.0, index=dataset.index, columns=dataset.columns)
        df_predict = pd.DataFrame(0, index=dataset.index, columns=dataset.columns)

        columns = dataset.columns if target_columns is None else target_columns

        for column in columns:
            if column_types[column] not in self.target_types:
                continue
            extraction_starttime = time.time()
            target = (
                df_detect[column].dropna()
                if self.detector_name == "MAD"
                else df_detect[column]
            )
            if len(target.dropna()) == 0:
                continue
            target_values = self._extract_features(
                target, column_types[column], embeddings
            )

            if self.detector_name == "MAD":
                target_values = (
                    target_values[:, 0].reshape(-1, 1)
                    if target_values.ndim > 1
                    else target_values.reshape(-1, 1)
                )
            times["extraction"] += time.time() - extraction_starttime

            target_idx = target.index

            fitting_starttime = time.time()
            # print(column, target_values.shape)
            # print(target_values)
            self.detector.fit(target_values)
            times["fitting"] += time.time() - fitting_starttime

            scoring_starttime = time.time()
            # scores = np.minimum(self.detector.decision_scores_, 10000)
            scores = self.detector.predict_proba(target_values)[:, 1]
            # scores[np.isnan(target_values[:, 0])] = 1.0  # assign 1 score to rows with NaN values
            if not np.all(scores == scores[0]):
                df_score.loc[target_idx, column] = scores

            predictions = self.detector.labels_
            assessed.append(column)
            df_predict.loc[target_idx, column] = predictions
            times["scoring"] += time.time() - scoring_starttime

        times["total"] = time.time() - total_starttime

        return df_score, df_predict.astype(int), times, assessed


class FeatureDetector(DMVDetector):
    def __init__(self, target_types: List[str] = ["numeric", "categorical"]):
        """
        Initialize the FeatureDetector with specific columns.

        Args:
            columns (list, optional): The columns to use for detection.
        """
        self.target_types = target_types

    def _extract_features(
        self, column: pd.Series, type: str, embeddings: Dict[str, pd.DataFrame]
    ) -> np.ndarray:
        """
        Extract features from a pandas Series for the detector.

        Args:
            column (pd.Series): The column to extract features from.

        Returns:
            np.ndarray: The extracted features.
        """
        scaler = MinMaxScaler()
        if type == "categorical":
            encoder = OneHotEncoder()
            encoded = encoder.fit_transform(column.to_numpy().reshape(-1, 1))
            return encoded
        elif type == "numeric":
            numeric_features = split_mixed_column(column, str(column.name))
            if numeric_features[f"{column.name}_null"].sum() == 0:
                numeric_features.drop(columns=[f"{column.name}_null"], inplace=True)
            if numeric_features[f"{column.name}_str"].sum() == 0:
                numeric_features.drop(columns=[f"{column.name}_str"], inplace=True)
            numeric_features[f"{column.name}_num"] = scaler.fit_transform(
                numeric_features[[f"{column.name}_num"]]
            )
            return numeric_features.values
        else:
            raise ValueError(f"Unsupported type for column {column.name}: {type}")

    def __call__(
        self,
        dataset: pd.DataFrame,
        column_types: Dict[str, COLUMN_TYPES],
        target_columns: List[str] | None = None,
        embeddings: Dict[str, pd.DataFrame] = {},
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], List[str]]:

        times: Dict[str, float] = {
            "extraction": 0,
            "fitting": 0,
            "scoring": 0,
        }

        total_starttime = time.time()
        assessed = []

        df_detect = dataset.copy()
        df_score = pd.DataFrame(0.0, index=dataset.index, columns=dataset.columns)
        df_predict = pd.DataFrame(0, index=dataset.index, columns=dataset.columns)

        columns = dataset.columns if target_columns is None else target_columns

        for column in columns:
            if column_types[column] not in self.target_types:
                continue
            extraction_starttime = time.time()
            target_values = self._extract_features(
                df_detect[column], column_types[column], embeddings
            )
            times["extraction"] += time.time() - extraction_starttime

            target = df_detect[column]
            target_idx = target.index

            scoring_starttime = time.time()

            df_score.loc[target_idx, column] = target_values

            threshold = np.quantile(target_values, 0.9)
            assessed.append(column)
            df_predict.loc[target_idx, column] = (target_values > threshold).astype(int)
            times["scoring"] += time.time() - scoring_starttime

        times["total"] = time.time() - total_starttime

        return df_score, df_predict.astype(int), times, assessed


# target types: all
class TypeOutlierDetector2(FeatureDetector):
    def _extract_features(
        self, column: pd.Series, type: str, embeddings: Dict[str, pd.DataFrame]
    ) -> np.ndarray:
        """
        Extract the string length features from a pandas Series.

        Args:
            column (pd.Series): The column to extract features from.

        Returns:
            np.ndarray: The extracted features.
        """
        features = np.zeros(len(column), dtype=int)
        if type == "numeric":
            values = pd.to_numeric(column, errors="coerce")
            return values.isna().astype(int).to_numpy()
        elif type == "date":
            values, _, _ = datetime_to_numeric(column)
            return values.isna().astype(int).to_numpy()
        else:
            values = pd.to_numeric(column, errors="coerce")
            return values.notna().astype(int).to_numpy()


class TypeOutlierDetector(FeatureDetector):
    def _extract_features(
        self, column: pd.Series, type: str, embeddings: Dict[str, pd.DataFrame]
    ) -> np.ndarray:
        """
        Extract the string length features from a pandas Series.

        Args:
            column (pd.Series): The column to extract features from.

        Returns:
            np.ndarray: The extracted features.
        """
        features = np.zeros(len(column), dtype=int)
        for row_idx, value in enumerate(column):
            if value is None or pd.isna(value):
                features[row_idx] = 0  # null
                continue
            try:
                float(value)
                features[row_idx] = 1  # numeric
                continue
            except:
                pass
            try:
                pd.to_datetime(value)
                features[row_idx] = 2  # date
                continue
            except:
                pass
            features[row_idx] = 3  # text & other
        return features


# target types: all
class LengthOutlierDetectorFeat(FeatureDetector):
    def _extract_features(
        self, column: pd.Series, type: str, embeddings: Dict[str, pd.DataFrame]
    ) -> np.ndarray:
        """
        Extract the string length features from a pandas Series.

        Args:
            column (pd.Series): The column to extract features from.

        Returns:
            np.ndarray: The extracted features.
        """
        lengths = np.array(
            [len(str(value)) if not pd.isna(value) else 0 for value in column]
        )  # .reshape(-1, 1)
        # lengths = MinMaxScaler().fit_transform(lengths)
        return lengths


class LengthOutlierDetectorDist(DistributionFitDetector):
    def _extract_features(self, column: pd.Series, type: str) -> np.ndarray:
        """
        Extract the string length features from a pandas Series.

        Args:
            column (pd.Series): The column to extract features from.

        Returns:
            np.ndarray: The extracted features.
        """
        lengths = np.array(
            [len(str(value)) if not pd.isna(value) else 0 for value in column]
        )  # .reshape(-1, 1)
        # lengths = MinMaxScaler().fit_transform(lengths)
        return lengths


class LengthOutlierDetectorOD(PyODDetector):
    def _extract_features(
        self, column: pd.Series, type: str, embeddings: Dict[str, pd.DataFrame]
    ) -> np.ndarray:
        """
        Extract the string length features from a pandas Series.

        Args:
            column (pd.Series): The column to extract features from.

        Returns:
            np.ndarray: The extracted features.
        """
        lengths = np.array(
            [len(str(value)) if not pd.isna(value) else 0 for value in column]
        ).reshape(-1, 1)
        if len(lengths) > 0:
            lengths = MinMaxScaler().fit_transform(lengths)
        return lengths


class FrequencyOutlierDetector(FeatureDetector):
    def _extract_features(
        self, column: pd.Series, type: str, embeddings: Dict[str, pd.DataFrame]
    ) -> np.ndarray:
        """
        Extract the string length features from a pandas Series.

        Args:
            column (pd.Series): The column to extract features from.

        Returns:
            np.ndarray: The extracted features.
        """
        value_counts = column.value_counts()
        frequencies = column.map(value_counts).fillna(0).to_numpy()
        return frequencies / column.notna().sum()


@cache_with_limit(maxsize=128)
def count_repeated_substrings(column, substring_length):
    """
    Count the maximum number of repeated substrings of a given length in a pandas Series.

    Args:
        column (pd.Series): The column to analyze.
        substring_length (int): The length of substrings to consider.

    Returns:
        np.ndarray: An array containing the maximum number of repeated substrings for each row.
    """
    features = np.zeros((len(column), 1), dtype=int)

    for row_idx, value in enumerate(column):
        if pd.isna(value):
            continue
        value_str = str(value)
        max_repeats = 1
        if len(value_str) >= substring_length:
            prev_sub = None
            count = 1
            for offset in range(substring_length):
                for i in range(
                    offset, len(value_str) - substring_length + 1, substring_length
                ):
                    sub = value_str[i : i + substring_length]
                    if sub == prev_sub:
                        count += 1
                    else:
                        count = 1
                        prev_sub = sub
                    if count > max_repeats:
                        max_repeats = count
        features[row_idx, 0] = max_repeats

    return features


# target types: all
class RepeatedSubstringOutlierDetectorFeat(FeatureDetector):
    def __init__(self, target_types: List[str], substring_length: int = 2):
        """
        Initialize the detector with a specific substring length.

        Args:
            substring_length (int): The length of substrings to consider for repetition.
        """
        self.target_types = target_types
        self.substring_length = substring_length

    def _extract_features(
        self, column: pd.Series, type: str, embeddings: Dict[str, pd.DataFrame]
    ) -> np.ndarray:
        """
        Extract features based on the maximum number of repeated substrings of a given length.

        Args:
            column (pd.Series): The column to extract features from.

        Returns:
            np.ndarray: The extracted features.
        """
        if type == "date":
            column = (
                column.astype(str)
                .str.replace(":", "")
                .str.replace("-", "")
                .str.replace(" ", "")
                .str.replace(".", "")
            )
        return count_repeated_substrings(column, self.substring_length).reshape(-1)


class RepeatedSubstringOutlierDetectorDist(DistributionFitDetector):
    def __init__(self, target_types: List[str], substring_length: int = 2):
        """
        Initialize the detector with a specific substring length.

        Args:
            substring_length (int): The length of substrings to consider for repetition.
        """
        super().__init__(target_types=target_types)
        self.substring_length = substring_length

    def _extract_features(self, column: pd.Series, type: str) -> np.ndarray:
        """
        Extract features based on the maximum number of repeated substrings of a given length.

        Args:
            column (pd.Series): The column to extract features from.

        Returns:
            np.ndarray: The extracted features.
        """
        if type == "date":
            column = (
                column.astype(str)
                .str.replace(":", "")
                .str.replace("-", "")
                .str.replace(" ", "")
                .str.replace(".", "")
            )
        return count_repeated_substrings(column, self.substring_length).reshape(-1)


class RepeatedSubstringOutlierDetectorOD(PyODDetector):
    def __init__(
        self, detector_name, target_types: List[str], substring_length: int = 2
    ):
        """
        Initialize the detector with a specific substring length.

        Args:
            substring_length (int): The length of substrings to consider for repetition.
        """
        super().__init__(detector_name=detector_name, target_types=target_types)
        self.substring_length = substring_length

    def _extract_features(
        self,
        column: pd.Series,
        type: str | None = None,
        embeddings: Dict[str, pd.DataFrame] | None = None,
    ) -> np.ndarray:
        """
        Extract features based on the maximum number of repeated substrings of a given length.

        Args:
            column (pd.Series): The column to extract features from.

        Returns:
            np.ndarray: The extracted features.
        """
        if type == "date":
            column = (
                column.astype(str)
                .str.replace(":", "")
                .str.replace("-", "")
                .str.replace(" ", "")
                .str.replace(".", "")
            )
        return count_repeated_substrings(column, self.substring_length)


@cache_with_limit(maxsize=128)
def get_key_distances(column):
    # Keyboard layout
    keyboard_rows = [
        ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-"],
        ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "[", "]"],
        ["a", "s", "d", "f", "g", "h", "j", "k", "l", ";", "'", "\\"],
        ["\\", "z", "x", "c", "v", "b", "n", "m", ",", ".", "/"],
    ]

    # Create map from character to (row, col) coordinates
    keyboard_map = {
        char: np.array([row_idx, col_idx])
        for row_idx, row in enumerate(keyboard_rows)
        for col_idx, char in enumerate(row)
    }

    def compute_avg_distance(value: str) -> float:
        if pd.isna(value):
            return 0.0
        value_str = str(value).replace(".", "")
        coords = [keyboard_map[char] for char in value_str if char in keyboard_map]
        if len(coords) < 2:
            return 0.0
        coords = np.stack(coords)
        diffs = coords[1:] - coords[:-1]
        dists = np.sqrt((diffs**2).sum(axis=1))
        return dists.mean()

    return column.map(compute_avg_distance).to_numpy()


# target types: all
class KeyDistanceOutlierDetectorFeat(FeatureDetector):
    def _extract_features(
        self, column: pd.Series, type: str, embeddings: Dict[str, pd.DataFrame]
    ) -> np.ndarray:
        """
        Extract features based on neighboring keyboard characters.

        Args:
            column (pd.Series): The column to extract features from.

        Returns:
            np.ndarray: The extracted features.
        """
        if type == "date":
            column = (
                column.astype(str)
                .str.replace(":", "")
                .str.replace("-", "")
                .str.replace(" ", "")
                .str.replace(".", "")
            )
        return -get_key_distances(column)


class KeyDistanceOutlierDetectorDist(DistributionFitDetector):
    def _extract_features(self, column: pd.Series, type: str) -> np.ndarray:
        """
        Extract features based on neighboring keyboard characters.

        Args:
            column (pd.Series): The column to extract features from.

        Returns:
            np.ndarray: The extracted features.
        """
        if type == "date":
            column = (
                column.astype(str)
                .str.replace(":", "")
                .str.replace("-", "")
                .str.replace(" ", "")
                .str.replace(".", "")
            )
        return get_key_distances(column)


class KeyDistanceOutlierDetectorOD(PyODDetector):
    def _extract_features(
        self, column: pd.Series, type: str, embeddings: Dict[str, pd.DataFrame]
    ) -> np.ndarray:
        """
        Extract features based on neighboring keyboard characters.

        Args:
            column (pd.Series): The column to extract features from.

        Returns:
            np.ndarray: The extracted features.
        """
        if type == "date":
            column = (
                column.astype(str)
                .str.replace(":", "")
                .str.replace("-", "")
                .str.replace(" ", "")
                .str.replace(".", "")
            )
        return get_key_distances(column).reshape(-1, 1)


@cache_with_limit(maxsize=128)
def get_uppercase_letters(column):
    def count_uppercase_letters(value: str) -> float:
        if pd.isna(value):
            return 0.0
        value_str = str(value)
        count = sum(1 for char in value_str if char.isupper())
        return count

    return column.map(count_uppercase_letters).to_numpy()


class CapitalLetterOutlierDetectorDist(DistributionFitDetector):
    def _extract_features(self, column: pd.Series, type: str) -> np.ndarray:
        """
        Extract features based on neighboring keyboard characters.

        Args:
            column (pd.Series): The column to extract features from.

        Returns:
            np.ndarray: The extracted features.
        """
        return get_uppercase_letters(column)


@cache_with_limit(maxsize=128)
def get_non_alphanumerical(column):
    def count_non_alphanumerical(value: str) -> float:
        if pd.isna(value):
            return 0.0
        value_str = str(value)
        count = sum(1 for char in value_str if not char.isalnum())
        return count

    return column.map(count_non_alphanumerical).to_numpy()


class NonAlphanumericalOutlierDetectorDist(DistributionFitDetector):
    def _extract_features(self, column: pd.Series, type: str) -> np.ndarray:
        """
        Extract features based on neighboring keyboard characters.

        Args:
            column (pd.Series): The column to extract features from.

        Returns:
            np.ndarray: The extracted features.
        """
        if type == "date":
            column = (
                column.astype(str)
                .str.replace(":", "")
                .str.replace("-", "")
                .str.replace(" ", "")
                .str.replace(".", "")
            )
        elif type == "numeric":
            column = column.astype(str).str.replace(".", "")
        return get_non_alphanumerical(column)


# target types: all
class SignOutlierDetectorFeat(FeatureDetector):
    def _extract_features(
        self, column: pd.Series, type: str, embeddings: Dict[str, pd.DataFrame]
    ) -> np.ndarray:
        """
        Extract features based on the maximum number of repeated substrings of a given length.

        Args:
            column (pd.Series): The column to extract features from.

        Returns:
            np.ndarray: The extracted features.
        """
        numeric_column = pd.to_numeric(column, errors="coerce").dropna()
        pos_fraction = (numeric_column.astype(float) > 0).sum() / len(numeric_column)
        neg_fraction = 1 - pos_fraction
        return (
            pd.to_numeric(column, errors="coerce")
            .map(
                lambda x: (
                    pos_fraction
                    if (not pd.isna(x) and float(x) > 0)
                    else (neg_fraction if (not pd.isna(x) and float(x) < 0) else 0)
                )
            )
            .to_numpy()
        )


# TODO: Mimic all other detectors using the querying capability of the LLM for comparison
# TODO: Use this for straight up DMV detection
class SemanticDetector(FeatureDetector):
    def __init__(
        self, LLM, target_types, targets: List[str] | Dict | None = None, invert=False
    ):
        """
        Initialize the SemanticDetector with a specific detector.

        Args:
            detector_name (str): The name of the detector to use.
            LLM: The language model to use for semantic detection. Has to implement the `embed` method.
            **kwargs: Additional keyword arguments to pass to the detector's constructor.
        """
        self.embedding_model = LLM
        self.invert = invert
        self.target_types = target_types
        self.targets = (
            targets
            if targets is not None
            else ["Missing", "Unknown", "Placeholder", "?"]
        )

    def _extract_features(
        self, column: pd.Series, type: str, embeddings: Dict[str, pd.DataFrame]
    ) -> np.ndarray:
        """
        Extract the string length features from a pandas Series.

        Args:
            column (pd.Series): The column to extract features from.

        Returns:
            np.ndarray: The extracted features.
        """
        # Encode the column values using the embedding model
        emb = embeddings[str(column.name)].astype(np.float32)
        # Normalize embeddings to unit vectors
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

        # Target embeddings
        target_emb = None
        if isinstance(self.targets, dict):
            if isinstance(self.targets[column.name][0], str):
                target_emb = np.array(
                    self.embedding_model.embed(self.targets[column.name]),
                    dtype=np.float32,
                )
            elif isinstance(self.targets[column.name][0], list) and isinstance(
                self.targets[column.name][0][0], float
            ):
                target_emb = np.array(self.targets[column.name], dtype=np.float32)
            else:
                raise ValueError(
                    f"Unsupported target format for column {column.name}: {self.targets[column.name]}"
                )
        else:
            if isinstance(self.targets[0], str):
                target_emb = np.array(
                    self.embedding_model.embed(self.targets), dtype=np.float32
                )
            elif isinstance(self.targets[0], list) and isinstance(
                self.targets[0][0], float
            ):
                target_emb = np.array(self.targets, dtype=np.float32)
            else:
                raise ValueError(f"Unsupported target format: {self.targets}")

        # Normalize embeddings to unit vectors
        target_emb = target_emb / (
            np.linalg.norm(target_emb, axis=1, keepdims=True) + 1e-8
        )

        # Compute similarities in batches to reduce memory usage
        batch_size = 1000
        n_samples = emb.shape[0]
        max_similarities = np.zeros(n_samples, dtype=np.float32)

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_emb = emb[i:end_idx]
            batch_similarities = batch_emb @ target_emb.T
            max_similarities[i:end_idx] = batch_similarities.max(axis=1)

        return max_similarities if not self.invert else 1 - max_similarities


class SemanticOutlierDetector(FeatureDetector):
    def __init__(self, LLM, target_types, num_neighbors=5):
        """
        Initialize the SemanticOutlierDetector with a specific detector.

        Args:
            detector_name (str): The name of the detector to use.
            LLM: The language model to use for semantic detection. Has to implement the `embed` method.
            **kwargs: Additional keyword arguments to pass to the detector's constructor.
        """
        self.embedding_model = LLM
        self.target_types = target_types
        self.num_neighbors = num_neighbors

    def _extract_features(
        self, column: pd.Series, type: str, embeddings: Dict[str, pd.DataFrame]
    ) -> np.ndarray:
        """
        Extract the string length features from a pandas Series.

        Args:
            column (pd.Series): The column to extract features from.

        Returns:
            np.ndarray: The extracted features.
        """
        # import psutil, os
        # process = psutil.Process(os.getpid())

        # Encode the column values using the embedding model
        emb = embeddings[str(column.name)].astype(np.float32)
        # Normalize embeddings to unit vectors
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

        # print(f"[MEMORY] Current memory usage (before similarities): {process.memory_info().rss / 1024 ** 2:.2f} MB")

        # Compute similarities in batches to reduce memory usage
        batch_size = 1000
        n_samples = emb.shape[0]
        outlier_scores = np.zeros(n_samples, dtype=np.float32)

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_emb = emb[i:end_idx]
            # Compute similarities for this batch against all embeddings
            batch_similarities = batch_emb @ emb.T
            # Sort to find top-k neighbors
            batch_similarities.sort(axis=1)
            # Compute outlier scores: 1 - mean of top num_neighbors similarities (excluding self)
            batch_outlier_scores = 1 - batch_similarities[
                :, -(self.num_neighbors + 1) : -1
            ].mean(axis=1)
            outlier_scores[i:end_idx] = batch_outlier_scores

        # print(f"[MEMORY] Current memory usage (after similarities): {process.memory_info().rss / 1024 ** 2:.2f} MB")

        outlier_scores = np.clip(outlier_scores, 0, 1)
        return outlier_scores


class SemanticOutlierDetectorNew(FeatureDetector):
    def __init__(self, LLM, target_types, num_neighbors=5):
        """
        Initialize the SemanticOutlierDetector with a specific detector.

        Args:
            detector_name (str): The name of the detector to use.
            LLM: The language model to use for semantic detection. Has to implement the `embed` method.
            **kwargs: Additional keyword arguments to pass to the detector's constructor.
        """
        self.embedding_model = LLM
        self.target_types = target_types
        self.num_neighbors = num_neighbors

    def _extract_features(
        self, column: pd.Series, type: str, embeddings: Dict[str, pd.DataFrame]
    ) -> np.ndarray:
        """
        Extract the string length features from a pandas Series.

        Args:
            column (pd.Series): The column to extract features from.

        Returns:
            np.ndarray: The extracted features.
        """
        # import psutil, os
        # process = psutil.Process(os.getpid())

        # Encode the column values using the embedding model
        emb = embeddings[str(column.name)].astype(np.float32)
        # Normalize embeddings to unit vectors
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

        # print(f"[MEMORY] Current memory usage (before similarities): {process.memory_info().rss / 1024 ** 2:.2f} MB")

        # Track None/NaN values in the original column
        null_mask = column.isna()

        # Compute similarities in batches to reduce memory usage
        batch_size = 10000000
        n_samples = emb.shape[0]
        outlier_scores = np.zeros(n_samples, dtype=np.float32)

        n_samples = emb.shape[0]

        M = max(16, min(48, int(np.log2(n_samples))))
        ef_construction = M * 4
        ef_search = int(ef_construction * 0.5)

        # Use IndexFlatIP for cosine similarity (inner product on normalized vectors)
        index = faiss.IndexHNSWFlat(emb.shape[1], M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        unique_emb, _ = np.unique(emb, axis=0, return_index=True)

        print("Unique embeddings:", unique_emb.shape[0], "out of", n_samples)

        # Ensure we don't search for more neighbors than available
        k = min(self.num_neighbors + 1, unique_emb.shape[0])

        if unique_emb.shape[0] > 1:
            index.add(unique_emb)

            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)

                # Search returns similarities (higher = more similar)
                similarities, _ = index.search(emb[i:end_idx], k)
                # print("Similarities shape:", similarities.shape)
                # print("Similarities min/max:", similarities.min(), similarities.max())
                # Convert to distances: distance = 1 - similarity
                # Skip first column (self-match) if k > 1
                if k > 1:
                    outlier_scores[i:end_idx] = 1 - similarities[:, 1:].mean(axis=1)
                else:
                    # If only 1 unique embedding, all are outliers
                    outlier_scores[i:end_idx] = 1.0

            outlier_scores = np.clip(outlier_scores, 0, 1)

        # Set scores to 0 for None/NaN values
        outlier_scores[null_mask] = 0

        print("min/max outlier scores:", outlier_scores.min(), outlier_scores.max())
        return outlier_scores


class SemanticOutlierDetectorNewDub(FeatureDetector):
    def __init__(self, LLM, target_types, num_neighbors=5):
        """
        Initialize the SemanticOutlierDetector with a specific detector.

        Args:
            detector_name (str): The name of the detector to use.
            LLM: The language model to use for semantic detection. Has to implement the `embed` method.
            **kwargs: Additional keyword arguments to pass to the detector's constructor.
        """
        self.embedding_model = LLM
        self.target_types = target_types
        self.num_neighbors = num_neighbors

    def _extract_features(
        self, column: pd.Series, type: str, embeddings: Dict[str, pd.DataFrame]
    ) -> np.ndarray:
        """
        Extract the string length features from a pandas Series.

        Args:
            column (pd.Series): The column to extract features from.

        Returns:
            np.ndarray: The extracted features.
        """
        # import psutil, os
        # process = psutil.Process(os.getpid())

        # Encode the column values using the embedding model
        emb = embeddings[str(column.name)].astype(np.float32)
        # Normalize embeddings to unit vectors
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

        # print(f"[MEMORY] Current memory usage (before similarities): {process.memory_info().rss / 1024 ** 2:.2f} MB")

        # Track None/NaN values in the original column
        null_mask = column.isna()

        # Compute similarities in batches to reduce memory usage
        batch_size = 10000000
        n_samples = emb.shape[0]
        outlier_scores = np.zeros(n_samples, dtype=np.float32)

        n_samples = emb.shape[0]

        M = max(16, min(48, int(np.log2(n_samples))))
        ef_construction = M * 4
        ef_search = int(ef_construction * 0.5)

        # Use IndexFlatIP for cosine similarity (inner product on normalized vectors)
        index = faiss.IndexHNSWFlat(emb.shape[1], M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search

        # Ensure we don't search for more neighbors than available
        k = min(self.num_neighbors + 1, emb.shape[0])

        if emb.shape[0] > 1:
            index.add(emb)

            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)

                # Search returns similarities (higher = more similar)
                similarities, _ = index.search(emb[i:end_idx], k)
                # print("Similarities shape:", similarities.shape)
                # print("Similarities min/max:", similarities.min(), similarities.max())
                # Convert to distances: distance = 1 - similarity
                # Skip first column (self-match) if k > 1
                if k > 1:
                    outlier_scores[i:end_idx] = 1 - similarities[:, 1:].mean(axis=1)
                else:
                    # If only 1 unique embedding, all are outliers
                    outlier_scores[i:end_idx] = 1.0

            outlier_scores = np.clip(outlier_scores, 0, 1)

        # Set scores to 0 for None/NaN values
        outlier_scores[null_mask] = 0

        print("min/max outlier scores:", outlier_scores.min(), outlier_scores.max())
        return outlier_scores


class MultiSemanticOutlierDetectorNew(DMVDetector):
    """
    Wrapper that runs SemanticOutlierDetectorNew with multiple num_neighbors settings efficiently.
    Returns results with suffixed names (e.g., 'semantic_outlier_k5', 'semantic_outlier_k10').
    """

    def __init__(
        self,
        LLM,
        target_types,
        num_neighbors_list=[3, 10, 25, 100],
        remove_duplicates=False,
    ):
        self.embedding_model = LLM
        self.target_types = target_types
        self.num_neighbors_list = sorted(num_neighbors_list)
        self.max_neighbors = max(num_neighbors_list)
        self.remove_duplicates = remove_duplicates

    def _extract_features_multi(
        self, column: pd.Series, type: str, embeddings: Dict[str, pd.DataFrame]
    ) -> Dict[int, np.ndarray]:
        """
        Extract features for multiple num_neighbors settings efficiently.
        Returns dict mapping num_neighbors -> scores.
        """
        emb = embeddings[str(column.name)].astype(np.float32)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

        null_mask = column.isna()
        n_samples = emb.shape[0]

        # Build index once
        M = max(16, min(48, int(np.log2(n_samples))))
        ef_construction = M * 4
        ef_search = int(ef_construction * 0.5)

        index = faiss.IndexHNSWFlat(emb.shape[1], M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search

        # Use unique embeddings if remove_duplicates is True
        if self.remove_duplicates:
            unique_emb, _ = np.unique(emb, axis=0, return_index=True)
            print(f"Unique embeddings: {unique_emb.shape[0]} out of {n_samples}")
            index_emb = unique_emb
        else:
            index_emb = emb

        # Query once with max_neighbors
        k = min(self.max_neighbors + 1, index_emb.shape[0])
        results = {}
        batch_size = 10000000

        if index_emb.shape[0] > 1:
            index.add(index_emb)

            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                similarities, _ = index.search(emb[i:end_idx], k)

                # Compute scores for each num_neighbors setting
                for num_neighbors in self.num_neighbors_list:
                    if num_neighbors not in results:
                        results[num_neighbors] = np.zeros(n_samples, dtype=np.float32)

                    k_actual = min(num_neighbors + 1, k)
                    if k_actual > 1:
                        results[num_neighbors][i:end_idx] = 1 - similarities[
                            :, 1:k_actual
                        ].mean(axis=1)
                    else:
                        results[num_neighbors][i:end_idx] = 1.0

            # Post-process all results
            for num_neighbors in self.num_neighbors_list:
                results[num_neighbors] = np.clip(results[num_neighbors], 0, 1)
                results[num_neighbors][null_mask] = 0
        else:
            # Handle edge case
            for num_neighbors in self.num_neighbors_list:
                results[num_neighbors] = np.zeros(n_samples, dtype=np.float32)

        return results

    def __call__(
        self,
        dataset: pd.DataFrame,
        column_types: Dict[str, COLUMN_TYPES],
        target_columns: List[str] | None = None,
        embeddings: Dict[str, pd.DataFrame] = {},
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], List[str]]]:
        """
        Returns a dictionary mapping detector names to (df_score, df_predict, times, assessed).
        """

        total_starttime = time.time()
        columns = dataset.columns if target_columns is None else target_columns

        # Initialize results dict for each configuration
        all_results = {}
        for num_neighbors in self.num_neighbors_list:
            df_score = pd.DataFrame(0.0, index=dataset.index, columns=dataset.columns)
            df_predict = pd.DataFrame(0, index=dataset.index, columns=dataset.columns)

            all_results[num_neighbors] = {
                "df_score": df_score,
                "df_predict": df_predict,
                "assessed": [],
            }

        # Process each column once for all configurations
        for column in columns:
            if column_types[column] not in self.target_types:
                continue

            if len(dataset[column].dropna()) == 0:
                continue

            # Get scores for all num_neighbors settings at once
            scores_dict = self._extract_features_multi(
                dataset[column], column_types[column], embeddings
            )

            # Distribute scores to respective dataframes
            for num_neighbors, scores in scores_dict.items():
                col_idx = all_results[num_neighbors]["df_score"].columns.get_loc(column)
                all_results[num_neighbors]["df_score"].iloc[:, col_idx] = scores

                # Simple threshold for predictions
                threshold = (
                    np.quantile(scores[scores > 0], 0.9) if (scores > 0).any() else 0.5
                )
                predictions = (scores > threshold).astype(int)
                all_results[num_neighbors]["df_predict"].iloc[:, col_idx] = predictions
                all_results[num_neighbors]["assessed"].append(column)

        total_time = time.time() - total_starttime

        # Package results with detector names
        final_results = {}
        for num_neighbors in self.num_neighbors_list:
            if self.remove_duplicates:
                detector_name = f"semantic_outlier_{num_neighbors}_new"
            else:
                detector_name = f"semantic_outlier_{num_neighbors}_new_dub"
            times = {"total": total_time / len(self.num_neighbors_list)}

            final_results[detector_name] = (
                all_results[num_neighbors]["df_score"],
                all_results[num_neighbors]["df_predict"],
                times,
                all_results[num_neighbors]["assessed"],
            )

        return final_results
