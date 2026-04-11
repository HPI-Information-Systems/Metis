import pandas as pd
import numpy as np
import time
from typing import Tuple, Dict, List

from .detector import DMVDetector
from .utils import force_numeric
from utils.datetime import datetime_to_numeric
import numpy as np

def mean_neighbor_diff2(x, n):
    x = np.asarray(x, dtype=float)
    N = len(x)

    diffs = []
    counts = np.zeros(N, dtype=int)

    for j in range(1, n+1):
        # valid indices for shift j
        # Calculate signed difference: positive if current value is MORE frequent than neighbor
        left_diff = x[j:] - x[:-j]  # x[i] - x[i-j]
        right_diff = x[:-j] - x[j:]  # x[i] - x[i+j]

        # insert into aligned positions
        left_idx = np.arange(j, N)
        right_idx = np.arange(0, N-j)

        # accumulate differences
        tmp = np.zeros(N, dtype=float)

        # left neighbors (i vs i-j): x[i] is MORE frequent if positive
        tmp[left_idx] += left_diff
        counts[left_idx] += 1

        # right neighbors (i vs i+j): x[i] is MORE frequent if positive
        tmp[right_idx] += right_diff
        counts[right_idx] += 1

        diffs.append(tmp)

    # sum over all shifts
    total_diffs = np.sum(diffs, axis=0)
    avg_diffs = np.divide(total_diffs, counts, out=np.zeros_like(total_diffs), where=counts>0)
    
    # Only return positive values (where current is more frequent than neighbors)
    # Negative values mean less frequent, so set them to 0
    return np.maximum(avg_diffs, 0)

def mean_neighbor_diff(x, n):
    x = np.asarray(x, dtype=float)
    N = len(x)

    diffs = []
    counts = np.zeros(N, dtype=int)

    for j in range(1, n+1):
        # valid indices for shift j
        # left shift: compare x[j:] with x[:-j]
        d = np.abs(x[j:] - x[:-j])

        # insert into aligned positions
        left_idx = np.arange(j, N)
        right_idx = np.arange(0, N-j)

        # accumulate differences
        tmp = np.zeros(N, dtype=float)

        # left neighbors (i vs i-j)
        tmp[left_idx] += d
        counts[left_idx] += 1

        # right neighbors (i vs i+j)
        tmp[right_idx] += d
        counts[right_idx] += 1

        diffs.append(tmp)

    # sum over all shifts
    return np.min(diffs, axis=0)
    total_diffs = np.sum(diffs, axis=0)
    return np.divide(total_diffs, counts, out=np.zeros_like(total_diffs), where=counts>0)

class FrequentValuesDetector(DMVDetector):
    def __init__(self, relative_prominence: float = 0.05, num_neighbors: int = 3, target_types: List[str] = ["numeric"]):
        """
        Initialize the FrequentValuesDetector with a specific detector.

        Args:
            relative_prominence (str): Relative prominence to count as DMV.
        """
        self.relative_prominence = relative_prominence
        self.num_neighbors = num_neighbors
        self.target_types = target_types

    def __call__(self, dataset: pd.DataFrame, types: Dict[str, str], target_columns: List[str] = None, embeddings: Dict[str, pd.DataFrame] = {}) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
        
        times = {
            'preprocessing': 0,
            'valuelist_building': 0,
            'scoring': 0,

        }
        total_starttime = time.time()
        assessed = []
    
        df_detect, df_score, df_predict = dataset.copy(), dataset.copy(), dataset.copy()
        # print(df_detect["Length"])
        df_score.loc[:, :] = 0 #1
        df_score = df_score.astype(float)
        df_predict.loc[:, :] = 0
        df_predict = df_predict.astype(int)

        if target_columns is None:
            target_columns = dataset.columns.tolist()

        for target_column in target_columns:
            print("Processing column:", target_column)
            # print(df_detect[target_column])
            # Skip non-numeric columns
            if types[target_column] not in self.target_types:
                continue

            preprocessing_starttime = time.time()
            # Check if the column is a date
            if types[target_column] == "date":
                df_detect[target_column], _, _ = datetime_to_numeric(df_detect[target_column])

            elif types[target_column] == "numeric":
                df_detect[target_column] = force_numeric(df_detect[target_column])

            else:
                continue
            
            # print(df_detect[target_column])
            target = df_detect[target_column].dropna()

            target_idx = df_detect[target_column].dropna().index

            ###
            #round_to = (np.percentile(target, 99) - np.percentile(target, 1)) / 100
            times['preprocessing'] += time.time() - preprocessing_starttime

            valuelist_building_starttime = time.time()
            ###
            #values = ((target / round_to).round() * round_to).value_counts().sort_index()
            values = target.value_counts().sort_index()
            value_range = values.max() - values.min()
            times['valuelist_building'] += time.time() - valuelist_building_starttime
            
            scoring_starttime = time.time()
            npvals = values.to_numpy()
            if len(npvals) < 2:
                continue
            is_peak = (npvals[1:-1] > (npvals[:-2] + self.relative_prominence*value_range)) & (npvals[1:-1] > (npvals[2:] + self.relative_prominence*value_range))
            first_peak = npvals[0] > (npvals[1] + self.relative_prominence*value_range)
            last_peak = npvals[-1] > (npvals[-2] + self.relative_prominence*value_range)
            is_peak = np.concatenate(([first_peak], is_peak, [last_peak]))
            detection_labels = df_detect[target_column].dropna().isin(values.index.to_numpy()[is_peak]).astype(int).to_numpy()

            differences = mean_neighbor_diff(npvals, self.num_neighbors) / (value_range + 1e-5)

            # Map every value in the rounded target to its corresponding difference score
            rounded_values = df_detect[target_column].dropna()
            value_to_diff = dict(zip(values.index.to_numpy(), differences))
            detection_scores = rounded_values.map(value_to_diff).to_numpy()

            df_score.iloc[target_idx, df_detect.columns.get_loc(target_column)] = detection_scores
            assessed.append(target_column)
            df_predict.iloc[target_idx, df_detect.columns.get_loc(target_column)] = detection_labels

            times['scoring'] += time.time() - scoring_starttime

        times['total'] = time.time() - total_starttime

        return df_score, df_predict.astype(int), times, assessed

class FrequentValuesDetector2(DMVDetector):
    def __init__(self, relative_prominence: float = 0.05, num_neighbors: int = 3, target_types: List[str] = ["numeric"], nanvalue=0):
        """
        Initialize the FrequentValuesDetector with a specific detector.

        Args:
            relative_prominence (str): Relative prominence to count as DMV.
        """
        self.relative_prominence = relative_prominence
        self.num_neighbors = num_neighbors
        self.target_types = target_types
        self.nanvalue = nanvalue

    def __call__(self, dataset: pd.DataFrame, types: Dict[str, str], target_columns: List[str] = None, embeddings: Dict[str, pd.DataFrame] = {}) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
        
        times = {
            'preprocessing': 0,
            'valuelist_building': 0,
            'scoring': 0,

        }
        total_starttime = time.time()
        assessed = []
    
        df_detect, df_score, df_predict = dataset.copy(), dataset.copy(), dataset.copy()
        # print(df_detect["Length"])
        df_score.loc[:, :] = self.nanvalue #1
        df_score = df_score.astype(float)
        df_predict.loc[:, :] = 0
        df_predict = df_predict.astype(int)

        if target_columns is None:
            target_columns = dataset.columns.tolist()

        for target_column in target_columns:
            print("Processing column:", target_column)
            # print(df_detect[target_column])
            # Skip non-numeric columns
            if types[target_column] not in self.target_types:
                continue

            preprocessing_starttime = time.time()
            # Check if the column is a date
            if types[target_column] == "date":
                df_detect[target_column], _, _ = datetime_to_numeric(df_detect[target_column])

            elif types[target_column] == "numeric":
                df_detect[target_column] = force_numeric(df_detect[target_column])

            else:
                continue
            
            # print(df_detect[target_column])
            target = df_detect[target_column].dropna()

            target_idx = df_detect[target_column].dropna().index

            ###
            #round_to = (np.percentile(target, 99) - np.percentile(target, 1)) / 100
            times['preprocessing'] += time.time() - preprocessing_starttime

            valuelist_building_starttime = time.time()
            ###
            #values = ((target / round_to).round() * round_to).value_counts().sort_index()
            values = target.value_counts().sort_index()
            value_range = values.max() - values.min()
            times['valuelist_building'] += time.time() - valuelist_building_starttime
            
            scoring_starttime = time.time()
            npvals = values.to_numpy()
            if len(npvals) < 2:
                continue
            is_peak = (npvals[1:-1] > (npvals[:-2] + self.relative_prominence*value_range)) & (npvals[1:-1] > (npvals[2:] + self.relative_prominence*value_range))
            first_peak = npvals[0] > (npvals[1] + self.relative_prominence*value_range)
            last_peak = npvals[-1] > (npvals[-2] + self.relative_prominence*value_range)
            is_peak = np.concatenate(([first_peak], is_peak, [last_peak]))
            detection_labels = df_detect[target_column].dropna().isin(values.index.to_numpy()[is_peak]).astype(int).to_numpy()

            differences = mean_neighbor_diff2(npvals, self.num_neighbors) / (value_range + 1e-5)

            # Map every value in the rounded target to its corresponding difference score
            rounded_values = df_detect[target_column].dropna()
            value_to_diff = dict(zip(values.index.to_numpy(), differences))
            detection_scores = rounded_values.map(value_to_diff).to_numpy()

            df_score.iloc[target_idx, df_detect.columns.get_loc(target_column)] = detection_scores
            assessed.append(target_column)
            df_predict.iloc[target_idx, df_detect.columns.get_loc(target_column)] = detection_labels

            times['scoring'] += time.time() - scoring_starttime

        times['total'] = time.time() - total_starttime

        return df_score, df_predict.astype(int), times, assessed