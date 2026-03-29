from typing import List, Literal

import pandas as pd

from metis.metric.completeness.completeness_nullRatio_config import (
    completeness_nullRatio_config,
)
from metis.metric.config import MetricConfig
from metis.metric.metric import Metric
from metis.utils.dq_dimension import DQDimension
from metis.utils.dq_granularity import DQGranularity
from metis.utils.result import DQResult


class completeness_nullRatio(Metric):
    _gui_requires_reference: bool = False
    _gui_config_required: bool = False
    _gui_callable_config: bool = False
    _gui_recommended_granularities: frozenset = frozenset({
        DQGranularity.ROW, DQGranularity.COLUMN, DQGranularity.TABLE,
    })
    _gui_description: str = (
        "Completeness as the ratio of non-null values. Configurable to report "
        "per cell, per row, per column, or as a single table-level score."
    )
    def assess(
        self,
        data: pd.DataFrame,
        metric_config: str | MetricConfig | None = None,
    ) -> List[DQResult]:
        """
        Assess the completeness of the data by calculating the ratio and count of null values on different granularities. The ratio of non-null values is stored as the completeness quality measurement, while the count of null values is stored in the explanation for better interpretability. The metric can be configured using `completeness_nullRatio_config` to calculate the completeness on column, row level, or table-level granularity.

        :param data: DataFrame to assess.
        :param metric_config: Optional configuration for the metric.
        :return: List of DQResult objects containing completeness results.
        """

        if metric_config is None:
            config = completeness_nullRatio_config()
        else:
            config = self.load_config(metric_config, completeness_nullRatio_config)

        na_mask = data.isna()

        completeness = (~na_mask).astype(int)
        null_count = na_mask.astype(int)

        if config.aggregation_axis is not None:
            mean_completeness = completeness.mean(axis=config.aggregation_axis)
            mean_null_count = null_count.sum(axis=config.aggregation_axis)

            if config.aggregate_all:
                table_completeness = mean_completeness.mean()
                table_null_count = mean_null_count.sum()

                return [
                    DQResult(
                        timestamp=pd.Timestamp.now(),
                        DQvalue=table_completeness,
                        DQdimension=DQDimension.COMPLETENESS,
                        DQmetric=self.__class__.__name__,
                        columnNames=data.columns.tolist(),
                        DQgranularity=DQGranularity.TABLE,
                        DQexplanation={
                            "null_count": float(table_null_count),
                        },
                    )
                ]

            return self.create_aggregated_results(
                mean_completeness,
                mean_null_count,
                config.aggregation_axis,
                data.columns.tolist(),
            )

        return self.create_flat_results(completeness, null_count)

    def create_aggregated_results(
        self,
        mean_completeness: pd.Series,
        mean_null_count: pd.Series,
        aggregation_axis: Literal["index", "columns"],
        columns: List[str],
    ) -> List[DQResult]:
        results = []
        for (index, completeness), null_count in zip(
            mean_completeness.items(), mean_null_count.values
        ):
            row_index = int(str(index)) if aggregation_axis == "columns" else None
            col_names = columns if aggregation_axis == "columns" else [str(index)]

            result = DQResult(
                timestamp=pd.Timestamp.now(),
                DQvalue=completeness,
                DQdimension=DQDimension.COMPLETENESS,
                DQmetric=self.__class__.__name__,
                columnNames=col_names,
                rowIndex=row_index,
                DQexplanation={"null_count": float(null_count)},
                DQgranularity=(
                    DQGranularity.ROW
                    if aggregation_axis == "columns"
                    else DQGranularity.COLUMN
                ),
            )
            results.append(result)

        return results

    def create_flat_results(
        self, completeness: pd.DataFrame, null_count: pd.DataFrame
    ) -> List[DQResult]:
        results = []
        for col in completeness.columns:
            for row_index, (completeness_value, null_count_value) in enumerate(
                zip(completeness[col].values, null_count[col].values)
            ):
                result = DQResult(
                    timestamp=pd.Timestamp.now(),
                    DQvalue=float(completeness_value),
                    DQdimension=DQDimension.COMPLETENESS,
                    DQmetric=self.__class__.__name__,
                    columnNames=[col],
                    rowIndex=row_index,
                    DQexplanation={"null_count": float(null_count_value)},
                    DQgranularity=DQGranularity.CELL,
                )
                results.append(result)
        return results
