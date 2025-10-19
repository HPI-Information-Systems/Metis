from math import sqrt
from typing import List, Union

import pandas as pd

from metis.metric.config import MetricConfig
from metis.metric.consistency.config import RuleConsistencyConfig
from metis.metric.metric import Metric
from metis.utils.dq_dimension import DQDimension
from metis.utils.result import DQResult


class RuleConsistency(Metric):
    def assess(
        self,
        data: pd.DataFrame,
        reference: Union[pd.DataFrame, None] = None,
        metric_config: str | None | MetricConfig = None,
    ) -> List[DQResult]:
        """
        Assess the consistency of the data by checking each value for the given rules.

        :param data: DataFrame to assess.
        :param metric_config: Optional configuration for the metric.
        :return: List of DQResult objects containing consistency results.
        """
        if metric_config is None:
            raise ValueError(
                "Metric configuration is required for rule-based consistency assessment."
            )
        if isinstance(metric_config, str):
            raise ValueError(
                "Metric configuration must be a RuleConsistencyConfig instance. JSON loading is not supported."
            )
        if not isinstance(metric_config, RuleConsistencyConfig):
            raise ValueError(
                "Metric configuration must be a RuleConsistencyConfig instance."
            )

        rules = metric_config.rules

        results: List[DQResult] = []
        total_rows = len(data)

        for col_name in data.columns:
            column_rules = rules.get(col_name, [])
            if len(column_rules) == 0:
                print(
                    f"No consistency rules defined for column '{col_name}'. Skipping."
                )
                continue

            max_violation = 0.0
            column_results: List[DQResult] = []

            for row_index in range(total_rows):
                degree_of_violation = sum(
                    rule(data.at[row_index, col_name]) for rule in column_rules
                )
                measurement = 1 / (1 + degree_of_violation)
                max_violation = max(max_violation, degree_of_violation)

                result = DQResult(
                    mesTime=pd.Timestamp.now(),
                    DQvalue=measurement,
                    DQdimension=DQDimension.CONSISTENCY,
                    DQmetric=self.__class__.__name__,
                    columnNames=[col_name],
                    rowIndex=row_index,
                )
                column_results.append(result)

            maximum_rules_coverage = 1 / (1 + max_violation)
            for result in column_results:
                certainty = sqrt(
                    (1 - result.DQvalue + maximum_rules_coverage)
                    * maximum_rules_coverage
                )
                result.DQannotations = {
                    "certainty": certainty,
                }

            results.extend(column_results)

        return results
