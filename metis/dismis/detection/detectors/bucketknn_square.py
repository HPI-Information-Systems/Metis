"""
BucketKNN Square Detector - Optimized with Numba

Requires Numba for optimal performance:
    pip install numba
"""
import pandas as pd
import numpy as np
import time
from typing import Tuple, Dict, List
from numba import jit, prange

from utils.datetime import datetime_to_numeric
from .detector import DMVDetector
from .utils import force_numeric


@jit(nopython=True, parallel=True, cache=True)
def compute_knn_scores_numba(sorted_vals, sorted_buckets, sorted_idx, k):
    """
    Numba-optimized KNN score computation using bucket boundaries.
    Exploits the fact that for sorted 1D data, k-nearest neighbors from 
    different buckets must be at bucket boundaries.
    
    Algorithm:
    1. Identify bucket boundaries (first and last k items per bucket)
    2. For each point, determine which boundary neighbors to use based on position
    3. Compute mean distance to those boundary neighbors
    
    Complexity: O(n) instead of O(n²)
    """
    n = len(sorted_vals)
    scores = np.zeros(n, dtype=np.float64)
    
    # Find bucket boundaries: for each bucket, store start and end indices
    unique_buckets = np.unique(sorted_buckets)
    num_buckets = len(unique_buckets)
    
    # For each bucket, store [start_idx, end_idx]
    bucket_ranges = np.zeros((num_buckets, 2), dtype=np.int64)
    for b_idx, bucket_id in enumerate(unique_buckets):
        mask = sorted_buckets == bucket_id
        indices = np.where(mask)[0]
        bucket_ranges[b_idx, 0] = indices[0]  # start
        bucket_ranges[b_idx, 1] = indices[-1]  # end
    
    # Process points in parallel
    for i in prange(n):
        current_bucket = sorted_buckets[i]
        current_value = sorted_vals[i]
        original_idx = sorted_idx[i]
        
        # Find which bucket we're in
        bucket_idx = -1
        for b_idx in range(num_buckets):
            if unique_buckets[b_idx] == current_bucket:
                bucket_idx = b_idx
                break
        
        if bucket_idx == -1:
            scores[original_idx] = np.nan
            continue
        
        bucket_start = bucket_ranges[bucket_idx, 0]
        bucket_end = bucket_ranges[bucket_idx, 1]
        bucket_size = bucket_end - bucket_start + 1
        position_in_bucket = i - bucket_start
        
        # Collect neighbors from adjacent buckets
        neighbors = np.zeros(k * 4, dtype=np.float64)  # Allow space for extended search
        neighbor_count = 0
        
        # Determine which adjacent bucket(s) to use
        has_prev = bucket_idx > 0
        has_next = bucket_idx < num_buckets - 1
        
        # Case 1: First or last bucket - only one adjacent bucket
        if not has_prev and has_next:
            # First bucket - use next bucket's first k items
            next_start = bucket_ranges[bucket_idx + 1, 0]
            for j in range(min(k, bucket_ranges[bucket_idx + 1, 1] - next_start + 1)):
                neighbors[neighbor_count] = abs(sorted_vals[next_start + j] - current_value)
                neighbor_count += 1
                
        elif has_prev and not has_next:
            # Last bucket - use previous bucket's last k items
            prev_end = bucket_ranges[bucket_idx - 1, 1]
            for j in range(min(k, prev_end - bucket_ranges[bucket_idx - 1, 0] + 1)):
                neighbors[neighbor_count] = abs(sorted_vals[prev_end - j] - current_value)
                neighbor_count += 1
                
        elif has_prev and has_next:
            # Middle bucket - determine based on position
            midpoint = bucket_size // 2
            
            if position_in_bucket < midpoint - k:
                # Close to start - use previous bucket's last k items
                prev_end = bucket_ranges[bucket_idx - 1, 1]
                for j in range(min(k, prev_end - bucket_ranges[bucket_idx - 1, 0] + 1)):
                    neighbors[neighbor_count] = abs(sorted_vals[prev_end - j] - current_value)
                    neighbor_count += 1
                    
            elif position_in_bucket > midpoint + k:
                # Close to end - use next bucket's first k items
                next_start = bucket_ranges[bucket_idx + 1, 0]
                for j in range(min(k, bucket_ranges[bucket_idx + 1, 1] - next_start + 1)):
                    neighbors[neighbor_count] = abs(sorted_vals[next_start + j] - current_value)
                    neighbor_count += 1
                    
            else:
                # In middle - use both adjacent buckets
                # Get from previous bucket (last k items)
                prev_end = bucket_ranges[bucket_idx - 1, 1]
                prev_available = min(k, prev_end - bucket_ranges[bucket_idx - 1, 0] + 1)
                for j in range(prev_available):
                    neighbors[neighbor_count] = abs(sorted_vals[prev_end - j] - current_value)
                    neighbor_count += 1
                
                # Get from next bucket (first k items)
                next_start = bucket_ranges[bucket_idx + 1, 0]
                next_available = min(k, bucket_ranges[bucket_idx + 1, 1] - next_start + 1)
                for j in range(next_available):
                    neighbors[neighbor_count] = abs(sorted_vals[next_start + j] - current_value)
                    neighbor_count += 1
        
        # Fallback: if we don't have enough neighbors, expand search radius
        if neighbor_count < k:
            radius = 2
            max_radius = max(bucket_idx, num_buckets - bucket_idx - 1)
            
            while neighbor_count < k and radius <= max_radius:
                # Search left (further back)
                if bucket_idx - radius >= 0:
                    prev_bucket_idx = bucket_idx - radius
                    prev_start = bucket_ranges[prev_bucket_idx, 0]
                    prev_end = bucket_ranges[prev_bucket_idx, 1]
                    prev_size = prev_end - prev_start + 1
                    
                    items_to_take = min(k, prev_size, k - neighbor_count)
                    for j in range(items_to_take):
                        neighbors[neighbor_count] = abs(sorted_vals[prev_end - j] - current_value)
                        neighbor_count += 1
                        if neighbor_count >= k:
                            break
                
                # Stop if we have enough neighbors
                if neighbor_count >= k:
                    break
                
                # Search right (further ahead)
                if bucket_idx + radius < num_buckets:
                    next_bucket_idx = bucket_idx + radius
                    next_start = bucket_ranges[next_bucket_idx, 0]
                    next_end = bucket_ranges[next_bucket_idx, 1]
                    next_size = next_end - next_start + 1
                    
                    items_to_take = min(k, next_size, k - neighbor_count)
                    for j in range(items_to_take):
                        neighbors[neighbor_count] = abs(sorted_vals[next_start + j] - current_value)
                        neighbor_count += 1
                        if neighbor_count >= k:
                            break
                
                # Stop if we have enough neighbors
                if neighbor_count >= k:
                    break
                
                radius += 1
        
        # If we collected neighbors, find the k closest
        if neighbor_count > 0:
            if neighbor_count <= k:
                # Use all neighbors if we have k or fewer
                scores[original_idx] = np.mean(neighbors[:neighbor_count])
            else:
                # Sort and take k closest
                sorted_neighbors = np.sort(neighbors[:neighbor_count])
                scores[original_idx] = np.mean(sorted_neighbors[:k])
        else:
            scores[original_idx] = np.nan
    
    return scores


class BucketKNN(DMVDetector):
    def __init__(self, num_buckets: int = 100, k: int = 5, target_types: List[str] = ["numeric", "date"]):
        """
        KNN-based outlier detector for 1D numeric data with bucket exclusion.

        Args:
            num_buckets (int): Number of buckets to split value range.
            k (int): Number of nearest neighbors to consider (outside own bucket).
            outlier_fraction (float): Fraction of highest scores to label as outliers.
        """
        self.num_buckets = num_buckets
        self.k = k
        self.gap_factor = 50.0  # factor for median gap to consider as outlier

    def __call__(
        self, 
        dataset: pd.DataFrame, 
        types: Dict[str, str], 
        target_columns: List[str] = None,
        embeddings: Dict[str, pd.DataFrame] = {}
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
        
        times = {
            'preprocessing': 0,
            'valuelist_building': 0,
            'scoring': 0,
        }
        total_starttime = time.time()
        assessed = []
    
        # More efficient DataFrame initialization - avoid full copies
        df_score = pd.DataFrame(1.0, index=dataset.index, columns=dataset.columns)
        df_predict = pd.DataFrame(0, index=dataset.index, columns=dataset.columns, dtype=int)
        
        # Work directly with the original dataset for detection
        df_detect = dataset

        if target_columns is None:
            target_columns = dataset.columns.tolist()

        for target_column in target_columns:
            preprocessing_starttime = time.time()
            
            # Work with column data directly to avoid unnecessary copies
            column_data = df_detect[target_column]
            
            # ensure float
            if types[target_column] == "date":
                processed_data, _, _ = datetime_to_numeric(column_data)
            elif types[target_column] == "numeric":
                processed_data = force_numeric(column_data)
            else:
                continue
            
            target = processed_data.dropna()
            print(target.to_list())
            if len(target) < self.k + 1:
                continue  # not enough data
            
            # Min-max normalization - compute min/max once
            target_min, target_max = target.min(), target.max()
            target_range = target_max - target_min + 1e-9
            target = (target - target_min) / target_range
            
            target_idx = target.index
            values = target.to_numpy(dtype=float)
            n = len(values)

            # Ensure we have at least k samples per bucket
            # Each bucket needs at least k samples for the algorithm to work
            max_buckets = max(1, n // self.k)
            num_buckets = min(self.num_buckets, max_buckets)
            
            # Bucket assignment - vectorized
            bucket_ids = np.clip(
                np.floor(values * num_buckets).astype(int), 
                0, num_buckets - 1
            )
            times['preprocessing'] += time.time() - preprocessing_starttime

            valuelist_building_starttime = time.time()
            # sort values for efficient neighbor search
            sorted_idx = np.argsort(values)
            sorted_vals = values[sorted_idx]
            sorted_buckets = bucket_ids[sorted_idx]
            times['valuelist_building'] += time.time() - valuelist_building_starttime

            scoring_starttime = time.time()
            
            # Use Numba-optimized sorted search
            print(f"Using Numba-optimized sorted search (n={n})")
            scores = compute_knn_scores_numba(sorted_vals, sorted_buckets, sorted_idx, self.k)

            # --- gap-based outlier detection (optimized) ---
            # Use argpartition instead of full sort for better performance
            valid_scores = scores[~np.isnan(scores)]
            if len(valid_scores) > 1:
                sorted_scores_idx = np.argsort(scores)
                sorted_scores = scores[sorted_scores_idx]
                
                # Remove NaN values from sorted scores for diff calculation
                valid_sorted_mask = ~np.isnan(sorted_scores)
                if np.sum(valid_sorted_mask) > 1:
                    valid_sorted_scores = sorted_scores[valid_sorted_mask]
                    valid_sorted_idx = sorted_scores_idx[valid_sorted_mask]
                    
                    diffs = np.diff(valid_sorted_scores)
                    max_gap = 0.2
                    
                    gap_mask = diffs > max_gap
                    if np.any(gap_mask):
                        # Find first gap that exceeds threshold
                        split_idx = np.argmax(gap_mask) + 1
                        outlier_indices = valid_sorted_idx[split_idx:]
                    else:
                        outlier_indices = np.array([], dtype=int)
                else:
                    outlier_indices = np.array([], dtype=int)
            else:
                outlier_indices = np.array([], dtype=int)

            # Efficient label assignment
            labels = np.zeros(n, dtype=int)
            if len(outlier_indices) > 0:
                labels[outlier_indices] = 1

            # Efficient assignment using .iloc for better performance
            df_score.loc[target_idx, target_column] = scores
            assessed.append(target_column)
            df_predict.loc[target_idx, target_column] = labels

            times['scoring'] += time.time() - scoring_starttime
            print("Scoring time for column", target_column, ":", time.time() - scoring_starttime, "seconds")

        times['total'] = time.time() - total_starttime
        return df_score, df_predict, times, assessed