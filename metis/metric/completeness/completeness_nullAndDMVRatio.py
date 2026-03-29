from typing import List, Literal

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
from metis.utils.dq_granularity import DQGranularity
from metis.utils.result import DQResult

IS_VALID_MARKER = 0
IS_NULL_MARKER = 1
IS_DMV_MARKER = 2


class completeness_nullAndDMVRatio(Metric):
    def assess(
        self,
        data: pd.DataFrame,
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

        completeness = (marked_cells == IS_VALID_MARKER).astype(int)
        certainty = self.certainty(marked_cells)

        if config.aggregation_axis is not None:
            mean_completeness = completeness.mean(axis=config.aggregation_axis)
            mean_certainty = certainty.mean(axis=config.aggregation_axis)

            if config.aggregate_all:
                table_completeness = mean_completeness.mean()
                table_certainty = mean_certainty.mean()
                return [
                    DQResult(
                        timestamp=pd.Timestamp.now(),
                        DQvalue=table_completeness,
                        DQdimension=DQDimension.COMPLETENESS,
                        DQmetric=self.__class__.__name__,
                        columnNames=data.columns.tolist(),
                        DQexplanation={"certainty": float(table_certainty)},
                        DQgranularity=DQGranularity.TABLE,
                    )
                ]

            return self.create_aggregated_results(
                mean_completeness,
                mean_certainty,
                config.aggregation_axis,
                data.columns.tolist(),
            )

        return self.create_flat_results(completeness, certainty)

    def certainty(self, marks: pd.DataFrame):
        # .replace with a dict sometimes throws an IndexError during pandas memory cleanup. Reason not yet identified, but using chained .replace calls seems to mitigate the issue.
        return (
            marks.replace(IS_VALID_MARKER, FAHES_RECALL)
            .replace(IS_NULL_MARKER, 1)
            .replace(IS_DMV_MARKER, FAHES_PRECISION)
        )

    def create_aggregated_results(
        self,
        mean_completeness: pd.Series,
        mean_certainty: pd.Series,
        aggregation_axis: Literal["index", "columns"],
        columns: List[str],
    ) -> List[DQResult]:
        results = []
        for (index, completeness), certainty in zip(
            mean_completeness.items(), mean_certainty.values
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
                DQexplanation={"certainty": float(certainty)},
                DQgranularity=(
                    DQGranularity.ROW
                    if aggregation_axis == "columns"
                    else DQGranularity.COLUMN
                ),
            )
            results.append(result)

        return results

    def create_flat_results(
        self, completeness: pd.DataFrame, certainty: pd.DataFrame
    ) -> List[DQResult]:
        results = []
        for col in completeness.columns:
            for row_index, (completeness_value, certainty_value) in enumerate(
                zip(completeness[col].values, certainty[col].values)
            ):
                result = DQResult(
                    timestamp=pd.Timestamp.now(),
                    DQvalue=float(completeness_value),
                    DQdimension=DQDimension.COMPLETENESS,
                    DQmetric=self.__class__.__name__,
                    columnNames=[col],
                    rowIndex=row_index,
                    DQexplanation={"certainty": float(certainty_value)},
                    DQgranularity=DQGranularity.CELL,
                )
                results.append(result)
        return results
