import pandas as pd
import numpy as np
from typing import List, Union

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from semhash import SemHash

from metis.metric.metric import Metric
from metis.metric.minimality.minimality_clustering_config import minimality_clustering_config
from metis.utils.dq_dimension import DQDimension
from metis.utils.result import DQResult

from metis.utils.similarity_measures.row import row_similarity

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

        if metric_config is None:
            raise ValueError(
                f"Metric configuration is required for metric {minimality_clustering_config.__name__} but None was provided."
            )

        config = self.load_config(metric_config, minimality_clustering_config)

        n_rows = len(data)

        if n_rows <= 1:
            minimality = 1.0
            num_clusters = n_rows
        else:
            if config.use_semhash:
                num_clusters = self._semhash_clusters(
                    data, config.similarity_threshold
                )
            else:
                num_clusters = self._custom_clusters(
                    data, config.similarity_threshold
                )

            minimality = (num_clusters - 1) / (n_rows - 1)

        result = DQResult(
            mesTime=pd.Timestamp.now(),
            DQdimension=DQDimension.MINIMALITY,
            DQmetric="Clustering",
            DQgranularity="table",
            DQvalue=float(minimality),
            DQexplanation={
                "total_rows": n_rows,
                "clusters": num_clusters,
                "use_semhash": config["use_semhash"],
                "similarity_threshold": config["similarity_threshold"],
            },
            columnNames=None,
            rowIndex=None,
            configJson=metric_config,
        )

        return [result]

    # ==================================================================
    # SemHash backend
    # ==================================================================

    def _semhash_clusters(self, data: pd.DataFrame, threshold: float) -> int:
        text_df = data.select_dtypes(include=["object"]).copy()

        if text_df.empty:
            return len(data)  # fallback: no text data → every data set own cluster

        records = text_df.astype(str).to_dict(orient="records")

        semhash = SemHash.from_records(
            records=records,
            columns=text_df.columns.tolist()
        )

        result = semhash.self_deduplicate(threshold=threshold)
        return len(result.selected)

    # ==================================================================
    # Custom similarity backend
    # ==================================================================

    def _custom_clusters(self, data: pd.DataFrame, threshold: float) -> int:
        df = data.select_dtypes(
            include=["object", "category", "number", "bool", "datetime64[ns]"]
        ).copy()

        n = len(df)
        sim_matrix = np.ones((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                sim = row_similarity(df.iloc[i], df.iloc[j])
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
