from typing import List

import pandas as pd

from metis.metric.completeness.completeness_nullRatio_config import (
    completeness_nullRatio_config,
)
from metis.metric.config import MetricConfig
from metis.metric.metric import Metric
from metis.utils.dq_dimension import DQDimension
from metis.utils.result import DQResult


class completeness_nullRatio(Metric):
    def assess(
        self,
        data: pd.DataFrame,
        reference: pd.DataFrame | None = None,
        metric_config: str | MetricConfig | None = None,
    ) -> List[DQResult]:
        """
        Assess the completeness of the data by calculating the ratio and count of null values on different granularities. The ratio of non-null values is stored as the completeness quality measurement, while the count of null values is stored in the explanation for better interpretability. The metric can be configured using `completeness_nullRatio_config` to calculate the completeness on column, row level, or table-level granularity.

        :param data: DataFrame to assess.
        :param reference: Optional reference DataFrame (not used in this metric).
        :param metric_config: Optional configuration for the metric.
        :return: List of DQResult objects containing completeness results.
        """

        config = self.load_config(metric_config, completeness_nullRatio_config)

        results = []

        na_mask = data.isna()

        def counts(null_mask: pd.Series):
            return null_mask.sum(), len(null_mask)

        def not_null_ratio(null_mask: pd.Series):
            null_count, total_count = counts(null_mask)
            return (total_count - null_count) / total_count

        def null_count(null_mask: pd.Series):
            null_count, _ = counts(null_mask)
            return null_count

        not_null_ratios = na_mask.agg(
            [not_null_ratio, null_count],
            axis=config.aggregation_axis,
        )

        if config.aggregation_axis == "index":
            not_null_ratios = not_null_ratios.T

        if config.aggregate_all:
            table_completeness = not_null_ratios["not_null_ratio"].mean()
            result = DQResult(
                mesTime=pd.Timestamp.now(),
                DQvalue=table_completeness,
                DQdimension=DQDimension.COMPLETENESS,
                DQmetric=self.__class__.__name__,
                columnNames=data.columns.tolist(),
                DQgranularity="table",
                DQexplanation={
                    "null_count": float(not_null_ratios["null_count"].sum()),
                }
            )
            results.append(result)
            return results

        for index, row in not_null_ratios.iterrows():
            row_index = (
                int(str(index)) if config.aggregation_axis == "columns" else None
            )
            col_names = (
                data.columns.tolist()
                if config.aggregation_axis == "columns"
                else [str(index)]
            )

            result = DQResult(
                mesTime=pd.Timestamp.now(),
                DQvalue=row["not_null_ratio"],
                DQdimension=DQDimension.COMPLETENESS,
                DQmetric=self.__class__.__name__,
                columnNames=col_names,
                rowIndex=row_index,
                DQgranularity=(
                    "row" if config.aggregation_axis == "columns" else "column"
                ),
                DQexplanation={
                    "null_count": float(row["null_count"]),
                }
            )
            results.append(result)

        return results
