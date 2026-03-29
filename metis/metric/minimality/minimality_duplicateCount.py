from typing import List, Union

import pandas as pd

from metis.metric.config import MetricConfig
from metis.metric.metric import Metric
from metis.utils.dq_dimension import DQDimension
from metis.utils.dq_granularity import DQGranularity
from metis.utils.result import DQResult


class minimality_duplicateCount(Metric):
    def assess(
        self,
        data: pd.DataFrame,
        metric_config: Union[MetricConfig, str, None] = None,
    ) -> List[DQResult]:
        """
        Assess the minimality for each attribute of a dataset by checking for unique values.

        :param data: DataFrame to assess.
        :param metric_config: Optional configuration for the metric.
        :return: List of DQResult objects containing completeness results.
        """
        results = []
        total_rows = len(data)

        for column in data.columns:
            # Count values that appear exactly once (not duplicated)
            unique_count = (~data[column].duplicated(keep=False)).sum()
            minimality = unique_count / total_rows if total_rows > 0 else 0

            # Attributes with 100% unique values are candidate keys
            annotations = {}
            if minimality == 1.0:
                annotations = {"CandidateKey": "CandidateKey"}

            result = DQResult(
                timestamp=pd.Timestamp.now(),
                DQdimension=DQDimension.MINIMALITY,
                DQmetric=self.__class__.__name__,
                DQgranularity=DQGranularity.COLUMN,
                DQvalue=minimality,
                DQexplanation=annotations,
                columnNames=[column],
            )
            results.append(result)

        return results
