from math import exp
from typing import List

import pandas as pd

from metis.metric.config import MetricConfig
from metis.metric.currency.config import CurrencyConfig
from metis.metric.metric import Metric
from metis.utils.dq_dimension import DQDimension
from metis.utils.result import DQResult


class Currency(Metric):
    def assess(
        self,
        data: pd.DataFrame,
        reference: pd.DataFrame | None = None,
        metric_config: str | MetricConfig | None = None,
    ) -> List[DQResult]:
        """
        Assess the currency of the data by calculating the deviation from the reference.

        :param data: DataFrame to assess.
        :param metric_config: Optional configuration for the metric.
        :return: List of DQResult objects containing currency results.
        """
        if not metric_config:
            raise ValueError(
                "Metric configuration is required for currency assessment."
            )

        config = self.load_config(metric_config, CurrencyConfig)

        ingestion_date_column = config.ingestion_date_column
        assessment_date = pd.to_datetime(
            config.simulated_assessment_date or pd.Timestamp.now()
        )

        results = []
        total_rows = len(data)
        decline_rate_variance = 0.1

        for col_name in data.columns:
            decline_rate = config.decline_rate_per_column.get(col_name)
            if decline_rate is None:
                print(
                    f"Decline rate for column '{col_name}' is not specified in the configuration. Skipping."
                )
                continue

            for row_index in range(total_rows):
                ingestion_date = pd.to_datetime(
                    str(data.at[row_index, ingestion_date_column]), dayfirst=True
                )
                delta = assessment_date - ingestion_date
                age = delta.days / 365
                measurement = exp(-decline_rate * age) if pd.notna(age) else 0
                certainty = 1 / (1 + decline_rate * decline_rate_variance * age)

                result = DQResult(
                    mesTime=pd.Timestamp.now(),
                    DQvalue=measurement,
                    DQdimension=DQDimension.CURRENCY,
                    DQmetric=self.__class__.__name__,
                    columnNames=[col_name],
                    rowIndex=row_index,
                    DQannotations={
                        "certainty": certainty,
                    },
                )
                results.append(result)

        return results
