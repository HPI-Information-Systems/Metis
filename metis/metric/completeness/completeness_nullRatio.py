from typing import List, Union

import pandas as pd

from metis.metric.config import MetricConfig
from metis.metric.metric import Metric
from metis.utils.dq_dimension import DQDimension
from metis.utils.result import DQResult


class completeness_nullRatio(Metric):
    def assess(self, data: pd.DataFrame, reference: Union[pd.DataFrame, None] = None, metric_config: Union[MetricConfig, str, None] = None) -> List[DQResult]:
        """
        Assess the completeness of the data by checking for null values.

        :param data: DataFrame to assess.
        :param reference: Optional reference DataFrame (not used in this metric).
        :param metric_config: Optional configuration for the metric.
        :return: List of DQResult objects containing completeness results.
        """
        results = []
        total_rows = len(data)

        for column in data.columns:
            null_count = data[column].isnull().sum()
            completeness = (total_rows - int(null_count)) / total_rows

            result = DQResult(
                mesTime=pd.Timestamp.now(),
                DQdimension=DQDimension.COMPLETENESS,
                DQmetric="NullRatio",
                DQgranularity="column",
                DQvalue=completeness,
                columnNames=[column],
            )
            results.append(result)

        return results
