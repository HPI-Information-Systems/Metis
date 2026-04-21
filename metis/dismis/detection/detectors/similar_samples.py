import time
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.preprocessing import MinMaxScaler

from metis.dismis.detection.detectors.detector import DMVDetector
from metis.dismis.detection.detectors.utils import encode_dataset
from metis.dismis.utils.types import COLUMN_TYPES


class SimilarSamplesDetector(DMVDetector):
    def __init__(self, num_neighbors: int = 10):
        """
        Initialize the SimilarSamplesDetector with a specific detector.

        Args:
            num_neighbors (str): Number of neighbors to use for the KNN detector.
        """
        self.num_neighbors = num_neighbors

    def __call__(
        self,
        dataset: pd.DataFrame,
        column_types: Dict[str, COLUMN_TYPES],
        target_columns: List[str] | None = None,
        embeddings: Dict[str, pd.DataFrame] = {},
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], List[str]]:

        times: Dict[str, float] = {
            "preprocessing": 0,
            "fitting": 0,
            "querying": 0,
            "scoring": 0,
        }
        total_starttime = time.time()
        assessed = []

        df_detect = dataset.copy()
        df_score = pd.DataFrame(0.0, index=dataset.index, columns=dataset.columns)
        df_predict = pd.DataFrame(0, index=dataset.index, columns=dataset.columns)

        scaler = MinMaxScaler()

        preprocessing_starttime = time.time()
        for idx, column in enumerate(df_detect.columns):
            if df_detect[column].dtype == "object":
                try:
                    df_detect[column] = scaler.fit_transform(
                        df_detect[column].astype(float).to_numpy().reshape(-1, 1)
                    )
                except (ValueError, TypeError):
                    df_detect[column] = df_detect[column].astype("category").cat.codes
            else:
                df_detect[column] = scaler.fit_transform(
                    df_detect[column].to_numpy().reshape(-1, 1)
                )
        times["preprocessing"] = time.time() - preprocessing_starttime

        columns = (
            df_detect.columns.tolist() if target_columns is None else target_columns
        )
        for target_column in columns:

            idx = df_detect.columns.tolist().index(target_column)

            target = df_detect.dropna()[target_column]
            target_idx = target.index
            target = target.to_numpy()

            fitting_starttime = time.time()
            tree = KDTree(
                df_detect.drop(columns=[target_column]).iloc[target_idx, :].values
            )
            times["fitting"] += time.time() - fitting_starttime

            querying_starttime = time.time()
            _, ind = tree.query(
                df_detect.drop(columns=[target_column]).iloc[target_idx, :],
                k=self.num_neighbors + 1,
            )
            times["querying"] += time.time() - querying_starttime

            scoring_starttime = time.time()
            differences = abs(
                target[ind][:, 1:]
                - np.repeat(target.reshape(-1, 1), self.num_neighbors, axis=1)
            )

            scores = np.mean(differences, axis=1)
            df_score.iloc[target_idx, idx] = scores
            assessed.append(target_column)

            predictions = (scores > np.percentile(scores, 99)).astype(
                int
            )  # Example threshold
            df_predict.iloc[target_idx, idx] = predictions
            times["scoring"] += time.time() - scoring_starttime

        times["total"] = time.time() - total_starttime

        return df_score, df_predict.astype(int), times, assessed


class FAISSSimilarSamplesDetector(DMVDetector):
    def __init__(
        self,
        num_neighbors: int = 10,
        use_correlations=False,
        target_types: List[str] = ["numeric", "text"],
        text_embedding_dim: int | None = 64,
        ohe_max_categories: int | None = 128,
        ohe_min_frequency: int | None = None,
    ):
        """
        Initialize the SimilarSamplesDetector with a specific detector.

        Args:
            num_neighbors (int): Number of neighbors to use for the FAISS KNN detector.
        """
        self.num_neighbors = num_neighbors
        self.use_correlations = use_correlations
        self.target_types = target_types
        self.text_embedding_dim = text_embedding_dim
        self.ohe_max_categories = ohe_max_categories
        self.ohe_min_frequency = ohe_min_frequency

    def __call__(
        self,
        dataset: pd.DataFrame,
        column_types: Dict[str, COLUMN_TYPES],
        target_columns: List[str] | None = None,
        embeddings: Dict[str, pd.DataFrame] = {},
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], List[str]]:

        times: Dict[str, float] = {
            "preprocessing": 0,
            "fitting": 0,
            "querying": 0,
            "scoring": 0,
        }

        total_starttime = time.time()
        assessed = []

        # Initialize score/predict DataFrames directly instead of copying dataset
        df_score = pd.DataFrame(0.0, columns=dataset.columns, index=dataset.index)
        df_predict = pd.DataFrame(0, columns=dataset.columns, index=dataset.index)

        # === Preprocessing ===
        preprocessing_starttime = time.time()
        df_detect = encode_dataset(
            dataset,  # Don't copy - encode_dataset handles it internally
            column_types,
            embeddings,
            text_embedding_dim=self.text_embedding_dim,
            ohe_max_categories=self.ohe_max_categories,
            ohe_min_frequency=self.ohe_min_frequency,
        )
        if self.use_correlations and not len(df_detect.columns) > 500:
            corr = df_detect.corr().abs()
            np.fill_diagonal(corr.values, 0)

        times["preprocessing"] = time.time() - preprocessing_starttime

        # === Detection Loop ===
        columns = (
            ["_".join(col.split("_")[:-1]) for col in df_detect.columns]
            if target_columns is None
            else target_columns
        )
        for target_column in columns:
            if column_types[target_column] not in self.target_types:
                continue
            # print(target_column)
            encoded_target_columns = [
                col
                for col in df_detect.columns
                if target_column == "_".join(col.split("_")[:-1])
            ]
            idx = dataset.columns.tolist().index(target_column)

            if self.use_correlations and not len(df_detect.columns) > 500:
                correlated_columns = []
                for col in encoded_target_columns:
                    median_corr = corr[col].mean()
                    for other_col, item in corr[col].items():
                        if (
                            abs(item) >= median_corr
                            and other_col not in encoded_target_columns
                        ):
                            correlated_columns.append(other_col)

            # print("Columns", encoded_target_columns, "correlated with", correlated_columns)
            # Get target array directly without intermediate DataFrame
            target_array = df_detect[
                encoded_target_columns
            ].values  # Already float32 from encode_dataset

            if self.use_correlations and not len(df_detect.columns) > 500:
                if len(correlated_columns) == 0:
                    # print(f"No correlated columns found for target column '{target_column}'. Skipping detection for this column.")
                    continue
                # Use column indexing to avoid creating intermediate DataFrame
                feature_cols = [
                    c for c in correlated_columns if c not in encoded_target_columns
                ]
                X = df_detect[feature_cols].values  # Already float32
            else:
                # Use boolean mask for column selection - more memory efficient than drop()
                feature_mask = ~df_detect.columns.isin(encoded_target_columns)
                X = df_detect.loc[:, feature_mask].values  # Already float32

            # === Fitting (Build FAISS index) ===
            fitting_starttime = time.time()

            n_samples = X.shape[0]

            M = max(16, min(48, int(np.log2(n_samples))))
            ef_construction = M * 4
            ef_search = int(ef_construction * 0.5)

            index = faiss.IndexHNSWFlat(X.shape[1], M)
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = ef_search
            index.add(X)
            times["fitting"] += time.time() - fitting_starttime

            # === Querying ===
            querying_starttime = time.time()
            distances, ind = index.search(
                X, self.num_neighbors + 1
            )  # +1 because the closest is the point itself
            times["querying"] += time.time() - querying_starttime
            # print(f"Queried in {time.time() - querying_starttime} seconds.")

            # === Scoring in batches ===
            scoring_starttime = time.time()
            batch_size = 1000
            n_batches = int(np.ceil(n_samples / batch_size))
            scores = np.zeros(n_samples, dtype=np.float32)
            predictions = np.zeros(n_samples, dtype=int)

            for batch_idx in range(n_batches):
                # print(f"[MEMORY] Current memory usage in similar samples batch {batch_idx}: {process.memory_info().rss / 1024 ** 2:.2f} MB")
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, n_samples)

                batch_ind = ind[start:end, 1:]
                batch_target = target_array[start:end]
                batch_distances = distances[start:end, 1:]

                # Compute scores more memory-efficiently
                n_target_dims = target_array.shape[1]
                if n_target_dims == 1:
                    # Single-column target: avoid repeat/tile overhead
                    neighbor_values = target_array[batch_ind, 0]  # shape: (batch, k)
                    differences = np.abs(neighbor_values - batch_target)  # broadcasting
                    dist_max = batch_distances.max()
                    if dist_max > 0:
                        neighbor_similarities = 1 - (batch_distances / dist_max)
                    else:
                        neighbor_similarities = np.ones_like(batch_distances)
                    batch_scores = (neighbor_similarities * differences).mean(axis=1)
                else:
                    # Multi-column target: reshape for efficient computation
                    neighbor_values = target_array[
                        batch_ind
                    ]  # shape: (batch, k, n_target_dims)
                    batch_target_exp = batch_target[
                        :, np.newaxis, :
                    ]  # shape: (batch, 1, n_target_dims)
                    differences = np.abs(
                        neighbor_values - batch_target_exp
                    )  # broadcasting, no tile needed

                    dist_max = batch_distances.max()
                    if dist_max > 0:
                        neighbor_similarities = 1 - (
                            batch_distances / dist_max
                        )  # shape: (batch, k)
                    else:
                        neighbor_similarities = np.ones_like(batch_distances)

                    # Weighted mean: expand similarities for broadcasting
                    weighted_diff = (
                        differences * neighbor_similarities[:, :, np.newaxis]
                    )
                    batch_scores = weighted_diff.mean(axis=(1, 2)).astype(np.float32)
                # scores = (differences).mean(axis=1)
                batch_scores[dataset[target_column].iloc[start:end].isnull()] = (
                    0  # Set scores to 0 for null values in the target column
                )
                scores[start:end] = batch_scores

                batch_predictions = (
                    batch_scores > np.percentile(batch_scores, 98)
                ).astype(int)
                predictions[start:end] = batch_predictions

            df_score.iloc[:, idx] = scores
            assessed.append(target_column)
            df_predict.iloc[:, idx] = predictions
            times["scoring"] += time.time() - scoring_starttime

        times["total"] = time.time() - total_starttime

        return df_score, df_predict.astype(int), times, assessed


class FAISSNoDubSimilarSamplesDetector(DMVDetector):
    def __init__(
        self,
        num_neighbors: int = 10,
        fetch_multiplier: int = 10,
        target_types: List[str] = ["numeric", "text"],
        text_embedding_dim: int | None = 64,
        ohe_max_categories: int | None = 128,
        ohe_min_frequency: int | None = None,
    ):
        """
        Initialize the FAISSSimilarSamplesDetector.

        Args:
            num_neighbors (int): Number of non-identical neighbors to use for scoring.
            fetch_multiplier (int): Multiplier to determine how many neighbors to fetch (e.g., 10x overfetch).
        """
        self.num_neighbors = num_neighbors
        self.fetch_multiplier = fetch_multiplier
        self.target_types = target_types
        self.text_embedding_dim = text_embedding_dim
        self.ohe_max_categories = ohe_max_categories
        self.ohe_min_frequency = ohe_min_frequency

    def __call__(
        self,
        dataset: pd.DataFrame,
        column_types: Dict[str, COLUMN_TYPES],
        target_columns: List[str] | None = None,
        embeddings: Dict[str, pd.DataFrame] = {},
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], List[str]]:

        times: Dict[str, float] = {
            "preprocessing": 0,
            "fitting": 0,
            "querying": 0,
            "scoring": 0,
        }

        total_starttime = time.time()

        # Initialize score/predict DataFrames directly instead of copying dataset
        df_score = pd.DataFrame(0.0, columns=dataset.columns, index=dataset.index)
        df_predict = pd.DataFrame(0, columns=dataset.columns, index=dataset.index)
        assessed = []

        # === Preprocessing ===
        preprocessing_starttime = time.time()
        df_detect = encode_dataset(
            dataset,  # Don't copy - encode_dataset handles it internally
            column_types,
            embeddings,
            text_embedding_dim=self.text_embedding_dim,
            ohe_max_categories=self.ohe_max_categories,
            ohe_min_frequency=self.ohe_min_frequency,
        )
        times["preprocessing"] = time.time() - preprocessing_starttime

        # === Detection Loop ===
        columns = (
            ["_".join(col.split("_")[:-1]) for col in df_detect.columns]
            if target_columns is None
            else target_columns
        )

        for target_column in columns:
            if column_types[target_column] not in self.target_types:
                continue
            encoded_target_columns = [
                col
                for col in df_detect.columns
                if target_column == "_".join(col.split("_")[:-1])
            ]
            idx = dataset.columns.tolist().index(target_column)

            # Get arrays directly without intermediate copies
            target_array = df_detect[encoded_target_columns].values  # Already float32
            feature_mask = ~df_detect.columns.isin(encoded_target_columns)
            X = df_detect.loc[
                :, feature_mask
            ].values  # Already float32, avoid drop() copy

            # === Fitting (Build FAISS index) ===
            fitting_starttime = time.time()
            n_samples = X.shape[0]
            M = max(16, min(48, int(np.log2(n_samples))))
            ef_construction = M * 4
            ef_search = int(ef_construction * 0.5)
            index = faiss.IndexHNSWFlat(X.shape[1], M)
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = ef_search
            index.add(X)
            times["fitting"] += time.time() - fitting_starttime

            # === Querying ===
            querying_starttime = time.time()
            fetch_k = (
                self.num_neighbors * self.fetch_multiplier
                if column_types[target_column] != "text"
                else self.num_neighbors * 2
            )
            distances, ind = index.search(
                X, fetch_k + 1
            )  # +1 because the closest is the point itself
            times["querying"] += time.time() - querying_starttime

            # === Scoring in batches ===
            scoring_starttime = time.time()
            batch_size = 1000
            n_batches = int(np.ceil(n_samples / batch_size))
            scores = np.zeros(n_samples, dtype=np.float32)
            predictions = np.zeros(n_samples, dtype=int)

            for batch_idx in range(n_batches):
                # print(f"Processing batch {batch_idx+1}/{n_batches} for column {target_column}")
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, n_samples)

                batch_ind = ind[start:end, 1:]
                batch_target = target_array[start:end]
                batch_distances = distances[start:end, 1:]

                neighbor_values = target_array[batch_ind].reshape(
                    batch_target.shape[0], -1
                )
                if len(batch_target.shape) == 1:
                    batch_target = batch_target.reshape(-1, 1)
                repeated_target = np.tile(batch_target, fetch_k)

                mask = (
                    target_array[batch_ind]
                    == repeated_target.reshape(
                        batch_target.shape[0], -1, batch_target.shape[1]
                    )
                ).sum(axis=2) != batch_target.shape[1]
                mask2 = np.full_like(mask, False, dtype=bool)
                for i, m in enumerate(mask):
                    valid_indices = np.flatnonzero(m)[: self.num_neighbors]
                    mask2[i, valid_indices] = True
                mask2 = np.repeat(mask2, batch_target.shape[1], axis=1)

                neighbor_distances = np.repeat(
                    batch_distances, batch_target.shape[1], axis=1
                )
                neighbor_similarities = 1 - neighbor_distances
                neighbor_similarities[~mask2] = 0
                neighbor_similarities_normalized = (
                    neighbor_similarities
                    / neighbor_similarities.sum(axis=1, keepdims=True)
                )
                differences = np.abs(neighbor_values - repeated_target)
                batch_scores = (neighbor_similarities_normalized * differences).sum(
                    axis=1
                )
                batch_scores[dataset[target_column].iloc[start:end].isnull()] = 0
                scores[start:end] = batch_scores
                batch_predictions = (
                    batch_scores > np.percentile(batch_scores, 98)
                ).astype(int)
                predictions[start:end] = batch_predictions

            df_score.iloc[:, idx] = scores
            assessed.append(target_column)
            df_predict.iloc[:, idx] = predictions
            times["scoring"] += time.time() - scoring_starttime

        times["total"] = time.time() - total_starttime

        return df_score, df_predict.astype(int), times, assessed


class MultiSimilarSamplesDetector(DMVDetector):
    """
    Efficient wrapper that runs similar samples detection with multiple configurations.
    Combines approximate_similar_samples, approximate_similar_samples_corr, and no_duplicate_similar_samples
    variants, building FAISS indices once per column instead of multiple times.
    """

    def __init__(
        self,
        num_neighbors_list=[5, 10, 25],
        include_correlations=True,
        include_no_duplicates=True,
        target_types: List[str] = ["numeric", "text"],
        text_embedding_dim: int | None = 64,
        ohe_max_categories: int | None = 256,
        ohe_min_frequency: int | None = None,
    ):
        """
        Args:
            num_neighbors_list: List of k values for neighbor counts
            include_correlations: If True, include correlation-based variants
            include_no_duplicates: If True, include duplicate-filtering variants
            target_types: Column types to process
        """
        self.num_neighbors_list = sorted(num_neighbors_list)
        self.max_neighbors = max(num_neighbors_list)
        self.include_correlations = include_correlations
        self.include_no_duplicates = include_no_duplicates
        self.target_types = target_types
        self.text_embedding_dim = text_embedding_dim
        self.ohe_max_categories = ohe_max_categories
        self.ohe_min_frequency = ohe_min_frequency
        self.fetch_multiplier = 10  # For no_duplicate variants

    def __call__(
        self,
        dataset: pd.DataFrame,
        column_types: Dict[str, COLUMN_TYPES],
        target_columns: List[str] | None = None,
        embeddings: Dict[str, pd.DataFrame] = {},
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], List[str]]]:

        total_starttime = time.time()

        # === Preprocessing ONCE ===
        preprocessing_starttime = time.time()
        df_detect = encode_dataset(
            dataset.copy(),
            column_types,
            embeddings,
            text_embedding_dim=self.text_embedding_dim,
            ohe_max_categories=self.ohe_max_categories,
            ohe_min_frequency=self.ohe_min_frequency,
        )

        # Initialize result dataframes
        df_score_template = pd.DataFrame(
            np.zeros(dataset.shape), columns=dataset.columns
        )
        df_predict_template = pd.DataFrame(
            np.zeros(dataset.shape, dtype=int), columns=dataset.columns
        )

        # Compute correlations ONCE if needed
        corr = None
        if self.include_correlations and len(df_detect.columns) <= 500:
            corr = df_detect.corr().abs()
            values = corr.values
            values.flags.writeable = True
            np.fill_diagonal(values, 0)
        preprocessing_time = time.time() - preprocessing_starttime

        # Initialize result containers for each configuration
        configs = self._initialize_configs(df_score_template, df_predict_template)

        # === Detection Loop - Process each column ONCE ===
        columns = (
            ["_".join(col.split("_")[:-1]) for col in df_detect.columns]
            if target_columns is None
            else target_columns
        )

        fitting_time = 0
        querying_time = 0
        scoring_time = 0

        for target_column in columns:
            if column_types[target_column] not in self.target_types:
                continue

            encoded_target_columns = [
                col
                for col in df_detect.columns
                if col.startswith(target_column + "_")
            ]
            idx = dataset.columns.tolist().index(target_column)

            target_array = df_detect[encoded_target_columns].to_numpy()

            # Determine correlated columns if needed
            correlated_columns = None
            if self.include_correlations and corr is not None:
                correlated_columns = []
                for col in encoded_target_columns:
                    median_corr = corr[col].mean()
                    for other_col, item in corr[col].items():
                        if (
                            abs(item) >= median_corr
                            and other_col not in encoded_target_columns
                        ):
                            correlated_columns.append(other_col)

                # Skip if no correlations found for correlation-based variants
                if len(correlated_columns) == 0:
                    correlated_columns = None

            # Build feature matrices
            X_all = df_detect.drop(columns=encoded_target_columns).values.astype(
                "float32"
            )
            X_corr = (
                df_detect.drop(columns=encoded_target_columns)[
                    correlated_columns
                ].values.astype("float32")
                if correlated_columns is not None
                else None
            )

            # === Build FAISS indices ONCE per column ===
            fitting_starttime = time.time()
            n_samples = X_all.shape[0]
            M = max(16, min(48, int(np.log2(n_samples))))
            ef_construction = M * 4
            ef_search = int(ef_construction * 0.5)

            # Index for non-correlation variants
            index_all = faiss.IndexHNSWFlat(X_all.shape[1], M)
            index_all.hnsw.efConstruction = ef_construction
            index_all.hnsw.efSearch = ef_search
            index_all.add(X_all)

            # Index for correlation variants (if applicable)
            index_corr = None
            if X_corr is not None:
                index_corr = faiss.IndexHNSWFlat(X_corr.shape[1], M)
                index_corr.hnsw.efConstruction = ef_construction
                index_corr.hnsw.efSearch = ef_search
                index_corr.add(X_corr)

            fitting_time += time.time() - fitting_starttime

            # === Query separately for each k to match exact FAISS behavior ===
            querying_starttime = time.time()

            # Pre-compute queries for each k value
            queries_all = {}  # k -> (distances, ind)
            queries_corr = {}  # k -> (distances, ind)
            queries_nodub = {}  # k -> (distances, ind)

            for k in self.num_neighbors_list:
                # Standard queries
                distances, ind = index_all.search(X_all, k + 1)
                queries_all[k] = (distances, ind)

                # Correlation queries
                if index_corr is not None:
                    distances, ind = index_corr.search(X_corr, k + 1)
                    queries_corr[k] = (distances, ind)

                # No-duplicate queries (need more neighbors)
                fetch_k = (
                    k * self.fetch_multiplier
                    if column_types[target_column] != "text"
                    else k * 2
                )
                distances, ind = index_all.search(X_all, fetch_k + 1)
                queries_nodub[k] = (distances, ind)

            querying_time += time.time() - querying_starttime

            # === Score all configurations ===
            scoring_starttime = time.time()

            for config in configs:
                # Skip correlation variants if no correlations found
                if config["use_corr"] and correlated_columns is None:
                    continue

                # Select appropriate distances and indices for this specific k
                k = config["k"]
                if config["use_corr"]:
                    distances, ind = queries_corr[k]
                elif config["remove_dups"]:
                    distances, ind = queries_nodub[k]
                else:
                    distances, ind = queries_all[k]

                # Compute scores for this configuration
                scores = self._compute_scores(
                    target_array=target_array,
                    distances=distances,
                    ind=ind,
                    num_neighbors=config["k"],
                    remove_duplicates=config["remove_dups"],
                    dataset=dataset,
                    target_column=target_column,
                    n_samples=n_samples,
                    fetch_k=ind.shape[1]
                    - 1,  # Pass actual fetch_k for no_duplicate variants
                )

                config["df_score"].iloc[:, idx] = scores
                threshold = (
                    np.percentile(scores, 98)
                    if config["remove_dups"]
                    else np.percentile(scores, 98)
                )
                config["df_predict"].iloc[:, idx] = (scores > threshold).astype(int)
                config["assessed"].append(target_column)

            scoring_time += time.time() - scoring_starttime

        # === Package results ===
        total_time = time.time() - total_starttime
        results = {}

        for config in configs:
            times = {
                "preprocessing": preprocessing_time,
                "fitting": fitting_time / len(configs),
                "querying": querying_time / len(configs),
                "scoring": scoring_time / len(configs),
                "total": total_time / len(configs),
            }

            results[config["name"]] = (
                config["df_score"],
                config["df_predict"],
                times,
                config["assessed"],
            )

        return results

    def _initialize_configs(self, df_score_template, df_predict_template):
        """Initialize result containers for each configuration."""
        configs = []

        for k in self.num_neighbors_list:
            # Standard approximate_similar_samples
            configs.append(
                {
                    "name": f"approximate_similar_samples_{k}",
                    "k": k,
                    "use_corr": False,
                    "remove_dups": False,
                    "df_score": df_score_template.copy(),
                    "df_predict": df_predict_template.copy(),
                    "assessed": [],
                }
            )

            # Correlation-based variant
            if self.include_correlations:
                configs.append(
                    {
                        "name": f"approximate_similar_samples_corr_{k}",
                        "k": k,
                        "use_corr": True,
                        "remove_dups": False,
                        "df_score": df_score_template.copy(),
                        "df_predict": df_predict_template.copy(),
                        "assessed": [],
                    }
                )

            # No-duplicate variant
            if self.include_no_duplicates:
                configs.append(
                    {
                        "name": f"no_duplicate_similar_samples_{k}",
                        "k": k,
                        "use_corr": False,
                        "remove_dups": True,
                        "df_score": df_score_template.copy(),
                        "df_predict": df_predict_template.copy(),
                        "assessed": [],
                    }
                )

        return configs

    def _compute_scores(
        self,
        target_array,
        distances,
        ind,
        num_neighbors,
        remove_duplicates,
        dataset,
        target_column,
        n_samples,
        fetch_k=None,
    ):
        """Compute scores for a specific configuration."""
        scores = np.zeros(n_samples, dtype=np.float32)
        batch_size = 1000
        n_batches = int(np.ceil(n_samples / batch_size))

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, n_samples)

            batch_ind = ind[start:end, 1:]  # Skip self
            batch_target = target_array[start:end]
            batch_distances = distances[start:end, 1:]

            if remove_duplicates:
                # Filter duplicates and get exactly num_neighbors unique neighbors
                # Use actual fetch_k from query, not num_neighbors
                actual_fetch_k = fetch_k if fetch_k is not None else batch_ind.shape[1]
                batch_scores = self._score_no_duplicates(
                    batch_target,
                    target_array,
                    batch_ind,
                    batch_distances,
                    num_neighbors,
                    actual_fetch_k,
                )
            else:
                # Standard scoring - already queried with exact k, no slicing needed
                batch_scores = self._score_standard(
                    batch_target, target_array, batch_ind, batch_distances
                )

            # Set scores to 0 for null values
            batch_scores[dataset[target_column].iloc[start:end].isnull()] = 0
            scores[start:end] = batch_scores

        return scores

    def _score_standard(self, batch_target, target_array, batch_ind, batch_distances):
        """Standard scoring (FAISSSimilarSamplesDetector logic)."""
        neighbor_values = target_array[batch_ind].reshape(batch_target.shape[0], -1)
        neighbor_distances = np.repeat(batch_distances, target_array.shape[1], axis=1)
        neighbor_similarities = 1 - (neighbor_distances / neighbor_distances.max())

        if len(batch_target.shape) == 1:
            batch_target = batch_target.reshape(-1, 1)
        repeated_target = np.tile(batch_target, batch_ind.shape[1])

        differences = np.abs(neighbor_values - repeated_target)
        return (neighbor_similarities * differences).mean(axis=1)

    def _score_no_duplicates(
        self,
        batch_target,
        target_array,
        batch_ind,
        batch_distances,
        num_neighbors,
        fetch_k,
    ):
        """No-duplicates scoring (FAISSNoDubSimilarSamplesDetector logic)."""
        if len(batch_target.shape) == 1:
            batch_target = batch_target.reshape(-1, 1)

        # Flatten for comparison (same as original)
        repeated_target = np.tile(batch_target, fetch_k)
        neighbor_values = target_array[batch_ind].reshape(batch_target.shape[0], -1)

        # Create mask for non-duplicate neighbors (matching original logic)
        # mask = True where neighbor != target (i.e., non-duplicates)
        repeated_target_reshaped = repeated_target.reshape(
            batch_target.shape[0], -1, batch_target.shape[1]
        )
        neighbor_values_reshaped = target_array[batch_ind]
        mask = (neighbor_values_reshaped == repeated_target_reshaped).sum(
            axis=2
        ) != batch_target.shape[1]

        # Select top num_neighbors unique neighbors per sample
        # mask2 will be True for the selected non-duplicate neighbors
        mask2 = np.full_like(mask, False, dtype=bool)
        for i, m in enumerate(mask):
            valid_indices = np.flatnonzero(m)[:num_neighbors]
            if len(valid_indices) > 0:
                mask2[i, valid_indices] = True

        # Expand mask for repeated dimensions
        mask2_expanded = np.repeat(mask2, batch_target.shape[1], axis=1)

        # Flatten arrays for scoring
        neighbor_distances_flat = np.repeat(
            batch_distances, target_array.shape[1], axis=1
        )

        # Calculate similarities - this matches the original exactly
        neighbor_similarities = 1 - neighbor_distances_flat
        # Original zeros out the SELECTED non-duplicates (seems counterintuitive but matches original)
        neighbor_similarities[~mask2_expanded] = 0

        # Normalize similarities
        neighbor_similarities_normalized = (
            neighbor_similarities / neighbor_similarities.sum(axis=1, keepdims=True)
        )

        # Calculate differences
        differences = np.abs(neighbor_values - repeated_target)

        return (neighbor_similarities_normalized * differences).sum(axis=1)
