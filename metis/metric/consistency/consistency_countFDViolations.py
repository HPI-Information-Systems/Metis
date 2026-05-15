import json
from typing import List, Union

import pandas as pd

from metis.metric.config import MetricConfig
from metis.metric.metric import Metric
from metis.utils.dq_dimension import DQDimension
from metis.utils.dq_granularity import DQGranularity
from metis.utils.result import DQResult


class consistency_countFDViolations(Metric):
    def assess(
        self,
        data: pd.DataFrame,
        reference: Union[pd.DataFrame, None] = None,
        metric_config: Union[MetricConfig, str, None] = None,
    ) -> List[DQResult]:
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

                # determinant values that violate FD 
                determinant_violating = grouped[grouped > 1].index

                # tuples that have a determinant that violates FD
                violating_tuples = data[data[determinant].isin(determinant_violating)]
                
                num_violating_tuples = violating_tuples.shape[0]
                total_rows = len(data[determinant])

                consistency = 1 - (num_violating_tuples / total_rows)

                violations = violating_tuples.to_json(orient="records") # returns json list like: [{"determinant":"value","dependent":"value"},..}]

                result = DQResult(
                    timestamp=pd.Timestamp.now(),
                    DQdimension=DQDimension.CONSISTENCY,
                    DQmetric=self.__class__.__name__,
                    DQgranularity=DQGranularity.TABLE,
                    DQvalue=consistency,
                    DQexplanation={f"{determinant}:{dependent}": violations}, 
                    columnNames=[determinant], # TODO: add determinant, dependent
                    configJson=metric_conf,
                )
                results.append(result)

        return results
