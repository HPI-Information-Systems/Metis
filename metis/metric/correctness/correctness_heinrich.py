from typing import List

import pandas as pd

from metis.metric.config import MetricConfig
from metis.metric.metric import Metric
from metis.utils.dq_dimension import DQDimension
from metis.utils.dq_granularity import DQGranularity
from metis.utils.numbers import clamp
from metis.utils.result import DQResult
from metis.utils.similarity_measures.string import levenshtein_distance


class correctness_heinrich(Metric):
    _gui_requires_reference: bool = True
    _gui_config_required: bool = False
    _gui_callable_config: bool = False
    _gui_cell_granularity: bool = True
    _gui_recommended_granularities: frozenset = frozenset({DQGranularity.CELL})
    _gui_description: str = (
        "Compares each cell against a reference DataFrame of the same shape. "
        "Numeric values use a normalized relative-distance score; strings use "
        "a normalized Levenshtein similarity. Produces a per-cell correctness "
        "value in `[0, 1]`."
    )
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

        if data.shape != reference.shape:
            raise ValueError(
                f"Data and reference must have the same shape for correctness assessment. Got data shape {data.shape} and reference shape {reference.shape}."
            )

        results = []
        total_rows = len(data)

        for col_name in data.columns:
            for row_index in range(total_rows):
                measurement = self.measure_correctness(
                    data[col_name].iat[row_index],
                    reference_value=reference[col_name].iat[row_index],
                    dtype=data[col_name].dtype,
                )

                result = DQResult(
                    timestamp=pd.Timestamp.now(),
                    DQvalue=measurement,
                    DQdimension=DQDimension.CORRECTNESS,
                    DQmetric=self.__class__.__name__,
                    columnNames=[col_name],
                    rowIndex=row_index,
                    DQgranularity=DQGranularity.CELL,
                )
                results.append(result)

        return results

    def measure_correctness(self, value, *, reference_value, dtype) -> float:
        if value == reference_value:
            return 1
        if pd.isna(value) or pd.isna(reference_value):
            return 0
        if pd.api.types.is_numeric_dtype(dtype):
            return clamp(
                1
                - abs(value - reference_value) / max(abs(reference_value), abs(value)),
                0,
                1,
            )
        if pd.api.types.is_string_dtype(dtype):
            max_len = max(len(str(value)), len(str(reference_value)))
            correctness = (
                1 - levenshtein_distance(str(value), str(reference_value)) / max_len
            )
            return correctness
        raise ValueError(
            f"Unsupported dtype for correctness measurement: {dtype} (value: {value}, reference_value: {reference_value})"
        )
