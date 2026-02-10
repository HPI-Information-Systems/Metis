from typing import List

import pandas as pd

from metis.metric.completeness.completeness_nullAndDMVRatio_config import (
    completeness_nullAndDMVRatio_config,
)
from metis.metric.config import MetricConfig
from metis.metric.metric import Metric
from metis.utils.disguised_missing_values.fahes.fahes import (
    FAHES_PRECISION,
    FAHES_RECALL,
    run_fahes,
)
from metis.utils.dq_dimension import DQDimension
from metis.utils.result import DQResult

IS_VALID_MARKER = 0
IS_NULL_MARKER = 1
IS_DMV_MARKER = 2


class completeness_nullAndDMVRatio(Metric):
    def assess(
        self,
        data: pd.DataFrame,
        reference: pd.DataFrame | None = None,
        metric_config: str | MetricConfig | None = None,
    ) -> List[DQResult]:
        """
        Assess the completeness of the data by checking for null values and disguised missing values. To detect disguised missing values, the FAHES algorithm by Qahtan et al. is applied to the data (paper: https://doi.org/10.1145/3219819.3220109). The completeness quality measurement is calculated as the ratio of valid values (non-null and non-disguised missing) to the total number of values. The metric can be configured using `completeness_nullAndDMVRatio_config` to calculate the completeness on column, row level, or table-level granularity.

        :param data: DataFrame to assess.
        :param reference: Optional reference DataFrame (not used in this metric).
        :param metric_config: Optional configuration for the metric.
        :return: List of DQResult objects containing completeness results.
        """

        config = self.load_config(metric_config, completeness_nullAndDMVRatio_config)

        results = []

        dmvs = run_fahes(data)
        self.logger.info(f"Detected DMVs:\n{dmvs}")

        marked_cells = pd.DataFrame(
            IS_VALID_MARKER, index=data.index, columns=data.columns
        )
        marked_cells[data.isna()] = IS_NULL_MARKER
        if dmvs is not None:
            for _, dmv_row in dmvs.iterrows():
                col = dmv_row["Attribute Name"]
                val = dmv_row["DMV"]
                marked_cells.loc[data[col] == val, col] = IS_DMV_MARKER

        def counts(marks: pd.Series):
            return (
                (marks == IS_NULL_MARKER).sum(),
                (marks == IS_DMV_MARKER).sum(),
                len(marks),
            )

        def completeness(marks: pd.Series):
            null_count, dmv_count, total_count = counts(marks)
            return (total_count - null_count - dmv_count) / total_count

        def certainty(marks: pd.Series):
            null_count, dmv_count, total_count = counts(marks)
            return self.certainty(null_count, dmv_count, total_count)

        aggregated_marks = marked_cells.agg(
            [completeness, certainty],
            axis=config.aggregation_axis,
        )

        if config.aggregation_axis == "index":
            aggregated_marks = aggregated_marks.T

        if config.aggregate_all:
            table_completeness = aggregated_marks["completeness"].mean()
            table_certainty = aggregated_marks["certainty"].mean()
            result = DQResult(
                mesTime=pd.Timestamp.now(),
                DQvalue=table_completeness,
                DQdimension=DQDimension.COMPLETENESS,
                DQmetric=self.__class__.__name__,
                columnNames=data.columns.tolist(),
                DQexplanation={"certainty": float(table_certainty)},
                DQgranularity="table",
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
                DQexplanation={"certainty": float(row["certainty"])},
                DQgranularity=(
                    "row" if config.aggregation_axis == "columns" else "column"
                ),
            )
            results.append(result)

        return results

    def certainty(self, null_count: int, dmv_count: int, total_count: int):
        minimum = (1 - FAHES_PRECISION) + (1 - FAHES_RECALL)
        return (
            1
            - (
                (1 - FAHES_PRECISION) * (dmv_count / total_count)
                + (1 - FAHES_RECALL) * (null_count / total_count)
            )
            / minimum
        )
