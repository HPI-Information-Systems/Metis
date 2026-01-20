from typing import List

import pandas as pd

from metis.metric.config import MetricConfig
from metis.metric.metric import Metric
from metis.utils.dq_dimension import DQDimension
from metis.utils.result import DQResult


class Completeness(Metric):
    def assess(
        self,
        data: pd.DataFrame,
        reference: pd.DataFrame | None = None,
        metric_config: str | MetricConfig | None = None,
    ) -> List[DQResult]:
        """
        Assess the completeness of the data by checking for missing values.

        :param data: DataFrame to assess.
        :param reference: Optional reference DataFrame (not used in this metric).
        :param metric_config: Optional configuration for the metric.
        :return: List of DQResult objects containing completeness results.
        """
        results = []
        total_rows = len(data)

        for column in data.columns:
            missing_count = data[column].isna().sum()
            completeness = (total_rows - int(missing_count)) / total_rows

            result = DQResult(
                mesTime=pd.Timestamp.now(),
                DQvalue=completeness,
                DQdimension=DQDimension.COMPLETENESS,
                DQmetric="Completeness",
                columnNames=[column],
            )
            results.append(result)

        return results
