import json
from typing import List, Union

import pandas as pd

from metis.metric.config import MetricConfig
from metis.metric.metric import Metric
from metis.utils.dq_dimension import DQDimension
from metis.utils.result import DQResult


class consistency_countFDViolations(Metric):
    def assess(self, data: pd.DataFrame, reference: Union[pd.DataFrame, None] = None, metric_config: Union[MetricConfig, str, None] = None) -> List[DQResult]:
        """
        Assess the consistency of a dataset by checking the compliance of a functional dependency specified in the metric_config.

        :param data: DataFrame to assess.
        :param metric_config: JSON that specifies FDs to check.
        :return: List of DQResult objects containing accuracy results.
        """
        results = []
        total_rows = len(data)

        if total_rows == 0:
            return results

        if metric_config is None:
            raise ValueError(
                "Metric configuration is required for consistency assessment."
            )
        if not isinstance(metric_config, str):
            raise ValueError(
                "Metric configuration must be a file path to a JSON configuration."
            )

        with open(metric_config, "r") as f:
            metric_conf = json.load(f)

        for determinant, dependents in metric_conf.items():
            if determinant not in data.columns:
                continue

            for dependent in dependents:
                if dependent not in data.columns:
                    continue

                # group by determinant and count unique dependent values
                grouped = data.groupby(determinant)[dependent].nunique()

                # find groups where there's more than one dependent value
                # for the same determinant (FD violation)
                violations = grouped[grouped > 1].index.tolist()

            consistency = 1 - (len(violations) / len(data[determinant]))

            result = DQResult(
                mesTime=pd.Timestamp.now(),
                DQdimension=DQDimension.CONSISTENCY,
                DQmetric="CountFDViolations",
                DQgranularity="table",
                DQvalue=consistency,
                DQexplanation={f"{determinant}:{dependent}": violations},  # FD
                columnNames=[determinant],
                configJson=metric_conf
            )
            results.append(result)

        return results
