from typing import List

import pandas as pd

from metis.metric.completeness.completeness_nullRate_config import (
    completeness_nullRate_config,
)
from metis.metric.config import MetricConfig
from metis.metric.metric import Metric
from metis.utils.dq_dimension import DQDimension
from metis.utils.result import DQResult


class completeness_nullRate(Metric):
    def assess(
        self,
        data: pd.DataFrame,
        reference: pd.DataFrame | None = None,
        metric_config: str | MetricConfig | None = None,
    ) -> List[DQResult]:
        """
        Assess the completeness of the data by checking for null values.

        :param data: DataFrame to assess.
        :param reference: Optional reference DataFrame (not used in this metric).
        :param metric_config: Optional configuration for the metric.
        :return: List of DQResult objects containing completeness results.
        """

        config = self.load_config(metric_config, completeness_nullRate_config)

        results = []

        na_mask = data.isna()

        def counts(marks: pd.Series):
            return marks.sum(), len(marks)

        def completeness(marks: pd.Series):
            null_count, total_count = counts(marks)
            return (total_count - null_count) / total_count

        aggregated_marks = na_mask.agg(
            [completeness],
            axis=config.aggregation_axis,
        )

        if config.aggregation_axis == "index":
            aggregated_marks = aggregated_marks.T

        if config.aggregate_all:
            table_completeness = aggregated_marks["completeness"].mean()
            result = DQResult(
                mesTime=pd.Timestamp.now(),
                DQvalue=table_completeness,
                DQdimension=DQDimension.COMPLETENESS,
                DQmetric=self.__class__.__name__,
                columnNames=data.columns.tolist(),
            )
            results.append(result)
            return results

        for index, row in aggregated_marks.iterrows():
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
                DQvalue=row["completeness"],
                DQdimension=DQDimension.COMPLETENESS,
                DQmetric=self.__class__.__name__,
                columnNames=col_names,
                rowIndex=row_index,
            )
            results.append(result)

        return results
