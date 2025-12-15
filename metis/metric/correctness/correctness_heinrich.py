from typing import List

import pandas as pd

from metis.metric.config import MetricConfig
from metis.metric.metric import Metric
from metis.utils.dq_dimension import DQDimension
from metis.utils.numbers import clamp
from metis.utils.result import DQResult
from metis.utils.strings import levenshtein_distance


class CorrectnessHeinrich(Metric):
    def assess(
        self,
        data: pd.DataFrame,
        reference: pd.DataFrame | None = None,
        metric_config: str | MetricConfig | None = None,
    ) -> List[DQResult]:
        """
        Assess the correctness of the data by calculating the deviation from the reference.

        :param data: DataFrame to assess.
        :param metric_config: Optional configuration for the metric.
        :return: List of DQResult objects containing correctness results.
        """
        if reference is None:
            raise ValueError(
                "Reference DataFrame is required for correctness assessment."
            )

        results = []
        total_rows = len(data)

        for col_name in data.columns:
            for row_index in range(total_rows):
                measurement = self.measure_correctness(
                    data.at[row_index, col_name],
                    reference_value=reference.at[row_index, col_name],
                    dtype=data[col_name].dtype,
                )

                result = DQResult(
                    mesTime=pd.Timestamp.now(),
                    DQvalue=measurement,
                    DQdimension=DQDimension.CORRECTNESS,
                    DQmetric=self.__class__.__name__,
                    columnNames=[col_name],
                    rowIndex=row_index,
                )
                results.append(result)

        return results

    def measure_correctness(self, value, *, reference_value, dtype) -> float:
        if value == reference_value:
            return 1
        if pd.isna(value) or pd.isna(reference_value):
            return 0
        if dtype == "int64" or dtype == "float64":
            return clamp(
                1
                - abs(value - reference_value) / max(abs(reference_value), abs(value)),
                0,
                1,
            )
        if dtype == "object":
            max_len = max(len(str(value)), len(str(reference_value)))
            correctness = (
                1 - levenshtein_distance(str(value), str(reference_value)) / max_len
            )
            return correctness
        raise ValueError(
            f"Unsupported dtype for correctness measurement: {dtype} (value: {value}, reference_value: {reference_value})"
        )
