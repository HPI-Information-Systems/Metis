from math import sqrt
from typing import Any, Callable, List, Union

import pandas as pd

from metis.metric.config import MetricConfig
from metis.metric.consistency.config import ConsistencyRuleBasedHinrichsConfig
from metis.metric.metric import Metric
from metis.utils.dq_dimension import DQDimension
from metis.utils.logging import logger as main_logger
from metis.utils.result import DQResult


class ConsistencyRuleBasedHinrichs(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.logger = main_logger.getChild(self.__class__.__name__)

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
                f"Metric configuration is required for metric {ConsistencyRuleBasedHinrichs.__name__} but None was provided."
            )
        if isinstance(metric_config, str):
            raise ValueError(
                f"Metric configuration must be an instance of {ConsistencyRuleBasedHinrichsConfig.__name__}. JSON loading is not supported."
            )
        if not isinstance(metric_config, ConsistencyRuleBasedHinrichsConfig):
            raise ValueError(
                f"Metric configuration must be an instance of {ConsistencyRuleBasedHinrichsConfig.__name__} but was of type {type(metric_config)}."
            )

        attribute_rules = metric_config.attribute_rules or {}
        tuple_rules = metric_config.tuple_rules or []

        results: List[DQResult] = []

        if tuple_rules:
            degree_of_violation: pd.Series[float] = data.apply(
                lambda x: self.sum_rules(tuple_rules, x), axis="columns"
            )

            dq_measurements = 1 / (1 + degree_of_violation)
            min_quality = dq_measurements.min()
            for row_index, dq_value in dq_measurements.items():
                certainty = sqrt(
                    (1 - dq_value + min_quality) * min_quality
                )

                results.append(
                    DQResult(
                        mesTime=pd.Timestamp.now(),
                        DQvalue=dq_value,
                        DQdimension=DQDimension.CONSISTENCY,
                        DQmetric=self.__class__.__name__,
                        columnNames=[],
                        rowIndex=int(str(row_index)),
                        DQannotations={
                            "certainty": certainty,
                        },
                    )
                )

        for col_name in data.columns:
            column_rules = attribute_rules.get(col_name, [])
            if not column_rules:
                self.logger.info(
                    f"No consistency rules defined for column '{col_name}'. Skipping."
                )
                continue

            degree_of_violation: pd.Series[float] = data[col_name].apply(
                lambda x: self.sum_rules(column_rules, x)
            )

            dq_measurements = 1 / (1 + degree_of_violation)
            min_quality = dq_measurements.min()

            for row_index, dq_value in dq_measurements.items():
                certainty = sqrt(
                    (1 - dq_value + min_quality) * min_quality
                )

                results.append(
                    DQResult(
                        mesTime=pd.Timestamp.now(),
                        DQvalue=dq_value,
                        DQdimension=DQDimension.CONSISTENCY,
                        DQmetric=self.__class__.__name__,
                        columnNames=[col_name],
                        rowIndex=int(str(row_index)),
                        DQannotations={
                            "certainty": certainty,
                        },
                    )
                )

        return results

    def sum_rules(self, rules: List[Callable], value: Any) -> float:
        return float(sum(rule(value) for rule in rules))
