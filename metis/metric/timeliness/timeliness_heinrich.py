from math import exp
from typing import List

import pandas as pd

from metis.metric.config import MetricConfig
from metis.metric.metric import Metric
from metis.metric.timeliness.timeliness_heinrich_config import (
    timeliness_heinrich_config,
)
from metis.utils.dq_dimension import DQDimension
from metis.utils.logging import logger as main_logger
from metis.utils.result import DQResult


class timeliness_heinrich(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.logger = main_logger.getChild(self.__class__.__name__)

    def assess(
        self,
        data: pd.DataFrame,
        reference: pd.DataFrame | None = None,
        metric_config: str | MetricConfig | None = None,
    ) -> List[DQResult]:
        """
        Assess the timeliness of the data by calculating how likely each cell is to be out of date based on a reference date and a decline rate. The reference date is either provided in the configuration or defaults to the current date.
        The formula used is: timeliness = exp(-decline_rate * age), where age and decline_rate are measured in years. The age is calculated as the difference between the reference date and the ingestion date of the tuple (defined by the ingestion_date_column in the configuration).

        :param data: DataFrame to assess.
        : param reference: Optional reference DataFrame (not used in this metric).
        :param metric_config: Configuration for the metric (required).
        :return: List of DQResult objects containing timeliness results.
        """
        if not metric_config:
            raise ValueError(
                "Metric configuration is required for timeliness assessment."
            )

        config = self.load_config(metric_config, timeliness_heinrich_config)

        ingestion_date_column = config.ingestion_date_column
        assessment_date = pd.to_datetime(
            config.simulated_assessment_date or pd.Timestamp.now()
        )

        results = []
        total_rows = len(data)

        for col_name in data.columns:
            decline_rate = config.decline_rate_per_column.get(col_name)
            if decline_rate is None:
                self.logger.info(
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

                result = DQResult(
                    mesTime=pd.Timestamp.now(),
                    DQvalue=measurement,
                    DQdimension=DQDimension.TIMELINESS,
                    DQmetric=self.__class__.__name__,
                    columnNames=[col_name],
                    rowIndex=row_index,
                )
                results.append(result)

        return results
