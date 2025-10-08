from typing import List, Union

import pandas as pd

from metis.metric.consistency.config import RuleConsistencyConfig
from metis.metric.metric import Metric
from metis.utils.result import DQResult


class RuleConsistency(Metric):
    def assess(
        self,
        data: pd.DataFrame,
        reference: Union[pd.DataFrame, None] = None,
        metric_config: str | None | RuleConsistencyConfig = None,
    ) -> List[DQResult]:
        """
        Assess the consistency of the data by checking each value for the given rules.

        :param data: DataFrame to assess.
        :param metric_config: Optional configuration for the metric.
        :return: List of DQResult objects containing consistency results.
        """
        if metric_config is None:
            raise ValueError("Metric configuration is required for rule-based consistency assessment.")
        if isinstance(metric_config, str):
            raise ValueError("Metric configuration must be a RuleConsistencyConfig instance. JSON loading is not supported.")

        rules = metric_config.rules

        results = []
        total_rows = len(data)

        for col_name in data.columns:
            column_rules = rules.get(col_name, [])
            for row_index in range(total_rows):
                degree_of_violation = sum(
                    rule(data.at[row_index, col_name]) for rule in column_rules
                )
                measurement = 1 / (1 + degree_of_violation)

                result = DQResult(
                    mesTime=pd.Timestamp.now(),
                    DQvalue=measurement,
                    DQdimension="Consistency",
                    DQmetric="RuleConsistency",
                    columnNames=[col_name],
                    rowIndex=row_index,
                )
                results.append(result)

        return results
