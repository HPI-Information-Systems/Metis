from typing import List, Union

import pandas as pd

from metis.metric.config import MetricConfig
from metis.metric.consistency.consistency_ruleBasedPipino_config import (
    consistency_ruleBasedPipino_config,
)
from metis.metric.metric import Metric
from metis.utils.dq_dimension import DQDimension
from metis.utils.logging import warn_unconfigured_columns
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
            fulfilled_rules_mask = pd.DataFrame(
                {
                    f"rule_{i}": data.apply(rule, axis="columns")
                    for i, rule in enumerate(tuple_rules)
                }
            )

            dq_measurements = fulfilled_rules_mask.sum(axis=1) / len(tuple_rules)
            certainties = self.certainties(fulfilled_rules_mask)
            for (row_index, dq_value), certainty in zip(
                dq_measurements.items(), certainties.values
            ):
                results.append(
                    self.create_result(
                        dq_value,
                        None,
                        int(str(row_index)),
                        float(certainty),
                    )
                )

        warn_unconfigured_columns(
            self.logger,
            set(data.columns),
            set(attribute_rules.keys()),
            "consistency rules",
        )

        for col_name in data.columns:
            column_rules = attribute_rules.get(col_name, [])
            if not column_rules:
                continue

            fulfilled_rules_mask = pd.DataFrame(
                {
                    f"rule_{i}": data[col_name].apply(rule)
                    for i, rule in enumerate(column_rules)
                }
            )

            dq_measurements = fulfilled_rules_mask.mean(axis=1)
            certainties = self.certainties(fulfilled_rules_mask)

            for (row_index, dq_value), certainty in zip(
                dq_measurements.items(), certainties.values
            ):
                results.append(
                    self.create_result(
                        dq_value,
                        col_name,
                        int(str(row_index)),
                        float(certainty),
                    )
                )

        return results

    def certainties(self, fulfilled_rules_mask: pd.DataFrame):
        rule_fulfillment_percentage = fulfilled_rules_mask.mean(axis=0)
        return (
            (1 - fulfilled_rules_mask - rule_fulfillment_percentage).abs().mean(axis=1)
        )

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
            DQexplanation={
                "certainty": certainty,
            },
            DQgranularity="cell" if col_name else "row",
        )
