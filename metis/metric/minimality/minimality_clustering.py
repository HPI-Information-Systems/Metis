import pandas as pd
import numpy as np
from typing import List

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from semhash import SemHash

from metis.metric.config import MetricConfig
from metis.metric.metric import Metric
from metis.metric.metric_meta import MetricMeta
from metis.metric.minimality.minimality_clustering_config import minimality_clustering_config
from metis.utils.dq_dimension import DQDimension
from metis.utils.dq_granularity import DQGranularity
from metis.utils.result import DQResult

from metis.utils.similarity_measures.row import row_similarity

class minimality_clustering(Metric):
    """
    Row-level minimality metric with configurable similarity backend.

    - Default: SemHash-based semantic deduplication
    - Optional: custom type-aware similarity + clustering
    """

    meta = MetricMeta(
        label="Clustering",
        description=(
            "Table-level minimality. Groups near-duplicate rows into clusters and "
            "reports `(clusters - 1) / (rows - 1)`, so 1.0 means every row is its "
            "own cluster and lower values mean more redundancy. The default "
            "backend is a type-aware row similarity with hierarchical clustering. "
            "Setting use_semhash switches to SemHash semantic deduplication."
        ),
        dimension=DQDimension.MINIMALITY,
        granularities=frozenset({DQGranularity.TABLE}),
        config_required=True,
    )

    def assess(
        self,
        data: pd.DataFrame,
        *,
        metric_config: str | MetricConfig | None = None,
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
            timestamp=pd.Timestamp.now(),
            DQdimension=DQDimension.MINIMALITY,
            DQmetric=self.__class__.__name__,
            DQgranularity=DQGranularity.TABLE,
            DQvalue=float(minimality),
            DQexplanation={
                "total_rows": n_rows,
                "clusters": num_clusters,
                "use_semhash": config.use_semhash,
                "similarity_threshold": config.similarity_threshold,
            },
            columnNames=None,
            rowIndex=None,
            configJson=config.to_json(),
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
