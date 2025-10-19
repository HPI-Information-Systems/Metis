from typing import List

import pandas as pd

from metis.metric.config import MetricConfig
from metis.metric.metric import Metric
from metis.utils.dq_dimension import DQDimension
from metis.utils.result import DQResult


class Correctness(Metric):
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
                    DQvalue=float(measurement),
                    DQdimension=DQDimension.CORRECTNESS,
                    DQmetric="Correctness",
                    columnNames=[col_name],
                    rowIndex=row_index,
                )
                results.append(result)

        return results

    def measure_correctness(self, value, *, reference_value, dtype):
        if value == reference_value:
            return 1
        if pd.isna(value) or pd.isna(reference_value):
            return 0
        if dtype == "int64" or dtype == "float64":
            return 1 - abs(value - reference_value) / max(abs(reference_value), abs(value))
        if dtype == "object":
            return 1 - self.levenshtein_distance(str(value), str(reference_value)) / max(
                len(str(value)), len(str(reference_value))
            )
        raise ValueError(
            f"Unsupported dtype for correctness measurement: {dtype} (value: {value}, reference_value: {reference_value})"
        )

    # https://stackoverflow.com/a/32558749
    def levenshtein_distance(self, s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(
                        1 + min((distances[i1], distances[i1 + 1], distances_[-1]))
                    )
            distances = distances_
        return distances[-1]
