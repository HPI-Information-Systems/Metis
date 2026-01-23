import json
import pandas as pd
import numpy as np
from typing import List, Union

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import Levenshtein

from semhash import SemHash

from metis.metric.metric import Metric
from metis.utils.result import DQResult


class minimality_clustering(Metric):
    """
    Row-level minimality metric with configurable similarity backend.

    - Default: SemHash-based semantic deduplication
    - Optional: custom type-aware similarity + clustering
    """

    def assess(
        self,
        data: pd.DataFrame,
        reference: Union[pd.DataFrame, None] = None,
        metric_config: Union[str, None] = None,
    ) -> List[DQResult]:

        # ---------------------------
        # Default configuration
        # ---------------------------
        config = {
            "use_semhash": True,
            "similarity_threshold": 0.85,
            "numeric_similarity": "normalized_distance",
            "timestamp_similarity": "normalized_distance",
            "boolean_similarity": "equality",
        }

        if metric_config:
            try:
                config.update(json.loads(metric_config))
            except Exception:
                pass

        n_rows = len(data)

        if n_rows <= 1:
            minimality = 1.0
            num_clusters = n_rows
        else:
            if config["use_semhash"]:
                num_clusters = self._semhash_clusters(
                    data, config["similarity_threshold"]
                )
            else:
                num_clusters = self._custom_clusters(
                    data, config["similarity_threshold"]
                )

            minimality = (num_clusters - 1) / (n_rows - 1)

        result = DQResult(
            mesTime=pd.Timestamp.now(),
            DQvalue=float(minimality),
            DQdimension="Minimality",
            DQmetric="clustering",
            columnNames=None,
            rowIndex=None,
            DQexplanation={
                "total_rows": n_rows,
                "clusters": num_clusters,
                "use_semhash": config["use_semhash"],
                "similarity_threshold": config["similarity_threshold"],
            },
            DQgranularity="table",
        )

        return [result]

    # ==================================================================
    # SemHash backend
    # ==================================================================

    def _semhash_clusters(self, data: pd.DataFrame, threshold: float) -> int:
        records = data.to_dict(orient="records")

        semhash = SemHash.from_records(
            records=records,
            columns=data.columns.tolist()
        )

        result = semhash.self_deduplicate(threshold=threshold)
        return len(result.selected)

    # ==================================================================
    # Custom similarity backend
    # ==================================================================

    def _custom_clusters(self, data: pd.DataFrame, threshold: float) -> int:
        df = self._select_supported_columns(data)

        n = len(df)
        sim_matrix = np.ones((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                sim = self._row_similarity(df.iloc[i], df.iloc[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        dist_matrix = 1.0 - sim_matrix
        condensed = squareform(dist_matrix, checks=False)
        linkage_matrix = linkage(condensed, method="single")

        labels = fcluster(
            linkage_matrix,
            t=1.0 - threshold,
            criterion="distance"
        )

        return len(set(labels))

    # ==================================================================
    # Similarity helpers
    # ==================================================================

    def _select_supported_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.select_dtypes(
            include=["object", "number", "bool", "datetime64[ns]"]
        ).copy()

    def _row_similarity(self, row_a: pd.Series, row_b: pd.Series) -> float:
        sims = []

        for col in row_a.index:
            a, b = row_a[col], row_b[col]

            if pd.isna(a) or pd.isna(b):
                continue

            if isinstance(a, str):
                sims.append(self._levenshtein_similarity(a, b))

            elif isinstance(a, (int, float)):
                sims.append(self._numeric_similarity(a, b))

            elif isinstance(a, bool):
                sims.append(1.0 if a == b else 0.0)

            elif isinstance(a, pd.Timestamp):
                sims.append(self._timestamp_similarity(a, b))

        return float(np.mean(sims)) if sims else 0.0

    @staticmethod
    def _levenshtein_similarity(a: str, b: str) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        d = Levenshtein.distance(a.lower(), b.lower())
        return 1.0 - d / max(len(a), len(b))

    @staticmethod
    def _numeric_similarity(a: float, b: float) -> float:
        denom = max(abs(a), abs(b), 1.0)
        return max(0.0, 1.0 - abs(a - b) / denom)

    @staticmethod
    def _timestamp_similarity(a: pd.Timestamp, b: pd.Timestamp) -> float:
        delta = abs((a - b).total_seconds())
        max_delta = max(abs(a.timestamp()), abs(b.timestamp()), 1.0)
        return max(0.0, 1.0 - delta / max_delta)
