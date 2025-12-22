from math import sqrt
from typing import Any, Callable, List, Union

import pandas as pd

from metis.metric.config import MetricConfig
from metis.metric.consistency.consistency_ruleBasedPipino_config import (
    consistency_ruleBasedPipino_config,
)
from metis.metric.metric import Metric
from metis.utils.dq_dimension import DQDimension
from metis.utils.result import DQResult


class consistency_ruleBasedPipino(Metric):
    def assess(
        self,
        data: pd.DataFrame,
        reference: Union[pd.DataFrame, None] = None,
        metric_config: str | None | MetricConfig = None,
    ) -> List[DQResult]:
        """
        Assess the consistency of the data by checking the given rules for each value. The rules are defined in the metric configuration. There are attribute rules that apply to individual columns and tuple rules that apply to entire rows. The quality measurement is calculated as 1 - degree_of_violation / N, where degree_of_violation is the sum of the result of all applicable rules for a given value/row and N is the total number of rules.
        Additionally, this metric assesses the certainty of the measurement based on the minimum quality in the assessed data. The certainty is calculated as sqrt((1 - dq_value) * (1 - min_quality)), where dq_value is the quality measurement for the specific value/row and min_quality is the lowest quality measurement observed in the dataset.

        :param data: DataFrame to assess.
        :param reference: Optional reference DataFrame (not used in this metric).
        :param metric_config: Mandatory configuration for the metric.
        :return: List of DQResult objects containing consistency results.
        """
        if metric_config is None:
            raise ValueError(
                f"Metric configuration is required for metric {consistency_ruleBasedPipino.__name__} but None was provided."
            )
        if isinstance(metric_config, str):
            raise ValueError(
                f"Metric configuration must be an instance of {consistency_ruleBasedPipino_config.__name__}. JSON loading is not supported."
            )
        if not isinstance(metric_config, consistency_ruleBasedPipino_config):
            raise ValueError(
                f"Metric configuration must be an instance of {consistency_ruleBasedPipino_config.__name__} but was of type {type(metric_config)}."
            )

        attribute_rules = metric_config.attribute_rules or {}
        tuple_rules = metric_config.tuple_rules or []

        results: List[DQResult] = []

        if tuple_rules:
            degree_of_violation: pd.Series[float] = data.apply(
                lambda x: self.sum_rules(tuple_rules, x), axis="columns"
            )

            dq_measurements = 1 - degree_of_violation / len(tuple_rules)
            min_quality = dq_measurements.min()
            for row_index, dq_value in dq_measurements.items():
                results.append(
                    self.create_result(
                        dq_value,
                        None,
                        int(str(row_index)),
                        self.certainty(dq_value, min_quality),
                    )
                )

        extraneous_rules = set(attribute_rules.keys()) - set(data.columns)
        if extraneous_rules:
            self.logger.warning(
                f"The following columns have consistency rules defined but are not present in the data: {extraneous_rules}. These rules will be ignored."
            )

        extraneous_columns = set(data.columns) - set(attribute_rules.keys())
        if extraneous_columns:
            self.logger.info(
                f"The following columns are present in the data but have no consistency rules defined: {extraneous_columns}. These columns will be skipped."
            )

        for col_name in data.columns:
            column_rules = attribute_rules.get(col_name, [])
            if not column_rules:
                continue

            degree_of_violation: pd.Series[float] = data[col_name].apply(
                lambda x: self.sum_rules(column_rules, x)
            )

            dq_measurements = 1 - degree_of_violation / len(column_rules)
            min_quality = dq_measurements.min()

            for row_index, dq_value in dq_measurements.items():
                results.append(
                    self.create_result(
                        dq_value,
                        col_name,
                        int(str(row_index)),
                        self.certainty(dq_value, min_quality),
                    )
                )

        return results

    def sum_rules(self, rules: List[Callable], value: Any) -> float:
        return float(sum(rule(value) for rule in rules))

    def certainty(self, dq_value: float, min_quality: float) -> float:
        return sqrt((1 - dq_value) * (1 - min_quality))

    def create_result(
        self, dq_value: float, col_name: str | None, row_index: int, certainty: float
    ) -> DQResult:
        return DQResult(
            mesTime=pd.Timestamp.now(),
            DQvalue=dq_value,
            DQdimension=DQDimension.CONSISTENCY,
            DQmetric=self.__class__.__name__,
            columnNames=[col_name] if col_name else [],
            rowIndex=row_index,
            DQannotations={
                "certainty": certainty,
            },
        )
