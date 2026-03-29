from typing import List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder

from metis.metric.config import MetricConfig
from metis.metric.correctness.correctness_heinrich_config import (
    correctness_heinrich_config,
)
from metis.metric.metric import Metric
from metis.utils.dq_dimension import DQDimension
from metis.utils.dq_granularity import DQGranularity
from metis.utils.numbers import clamp
from metis.utils.result import DQResult
from metis.utils.similarity_measures.levenshtein_distance import levenshtein_distance


class correctness_heinrich(Metric):
    def assess(
        self,
        data: pd.DataFrame,
        metric_config: str | MetricConfig | None = None,
    ) -> List[DQResult]:
        """
        Assess the correctness of the data by calculating the deviation from the reference.

        :param data: DataFrame to assess.
        :param metric_config: Optional configuration for the metric.
        :return: List of DQResult objects containing correctness results.
        """
        config = self.load_config(metric_config, correctness_heinrich_config)
        reference_data = pd.read_csv(config.reference_file_path)

        if data.shape != reference_data.shape:
            raise ValueError(
                f"Data and reference must have the same shape for correctness assessment. Got data shape {data.shape} and reference shape {reference_data.shape}."
            )

        results = []
        total_rows = len(data)

        for col_name in data.columns:
            for row_index in range(total_rows):
                measurement = self.measure_correctness(
                    data[col_name].iat[row_index],
                    reference_value=reference_data[col_name].iat[row_index],
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
