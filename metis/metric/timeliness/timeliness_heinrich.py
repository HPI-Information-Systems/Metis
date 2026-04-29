from math import exp, floor
from typing import List

import numpy as np
import pandas as pd

from metis.metric.config import MetricConfig
from metis.metric.metric import Metric
from metis.metric.timeliness.timeliness_heinrich_config import (
    timeliness_heinrich_config,
)
from metis.utils.datetime.datetime_precision import determine_datetime_precision
from metis.utils.dq_dimension import DQDimension
from metis.utils.dq_granularity import DQGranularity
from metis.utils.logging import warn_unconfigured_columns
from metis.utils.result import DQResult


class timeliness_heinrich(Metric):
    def assess(
        self,
        data: pd.DataFrame,
        metric_config: str | MetricConfig | None = None,
    ) -> List[DQResult]:
        """
        Assess the timeliness of the data by calculating how likely each cell is to be out of date based on a reference date and a decline rate. The reference date is either provided in the configuration or defaults to the current date.
        The formula used is: timeliness = exp(-decline_rate * age), where age and decline_rate are measured in years. The age is calculated as the difference between the reference date and the ingestion date of the tuple (defined by the ingestion_date_column in the configuration).

        :param data: DataFrame to assess.
        :param reference: Optional reference DataFrame (not used in this metric).
        :param metric_config: Configuration for the metric (required).
        :return: List of DQResult objects containing timeliness results.
        """
        if not metric_config:
            raise ValueError(
                "Metric configuration is required for timeliness assessment."
            )

        config = self.load_config(metric_config, timeliness_heinrich_config)
        results = []
        warn_unconfigured_columns(
            self.logger,
            set(data.columns),
            set(config.timeliness_config_per_column.keys()),
            "timeliness configuration",
        )

        for col_name, col_config in config.timeliness_config_per_column.items():
            if col_name not in data.columns:
                continue

            ingestion_date_column = col_config.ingestion_date_column
            assessment_date = pd.to_datetime(
                col_config.simulated_assessment_date or pd.Timestamp.now()
            )

            if not ingestion_date_column or ingestion_date_column not in data.columns:
                self.logger.warning(
                    f"Ingestion date column '{ingestion_date_column}' is not present in the data. Skipping assessment for column '{col_name}'."
                )
                return results

            ingestion_dates = pd.to_datetime(
                data[ingestion_date_column], **(col_config.to_datetime_kwargs or {})
            )
            ages_in_days = (
                (assessment_date - ingestion_dates).dt.total_seconds() / 60 / 60 / 24
            )
            precision_of_dates = data[ingestion_date_column].apply(
                determine_datetime_precision
            )
            age_and_precision = pd.DataFrame(
                {"age": ages_in_days, "precision": precision_of_dates}
            )

            decline_rate = col_config.decline_rate
            timeliness = pd.Series(np.exp(-decline_rate * ages_in_days))
            certainty = age_and_precision.apply(
                lambda row: self.certainty(
                    row["age"],
                    decline_rate or 0,
                    row["precision"],
                ),
                axis=1,
            )
            for row_index, (timeliness_value, certainty_value, age_and_precision_value) in enumerate(
                zip(timeliness.values, certainty.values, age_and_precision.values)
            ):
                result = DQResult(
                    timestamp=pd.Timestamp.now(),
                    DQvalue=timeliness_value,
                    DQdimension=DQDimension.TIMELINESS,
                    DQmetric=self.__class__.__name__,
                    columnNames=[col_name],
                    rowIndex=row_index,
                    DQexplanation={
                        "certainty": float(certainty_value),
                        "age_in_days": float(age_and_precision_value[0]),
                        "precision": age_and_precision_value[1],
                    },
                    DQgranularity=DQGranularity.CELL,
                )
                results.append(result)

        return results

    def certainty(self, age: float, decline_rate: float, precision: str) -> float:
        """
        Calculate the certainty of the timeliness measurement based on age, decline rate, and datetime precision.

        :param age: The age of the data in days.
        :param decline_rate: The decline rate per day.
        :param precision: The precision of the datetime ('year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond').
        :return: The certainty of the measurement.
        """
        lower_age_bound, upper_age_bound = self.age_precision_bounds(age, precision)
        # max_quality_difference = abs(exp(-decline_rate) - 1)
        unscaled_difference = abs(
            exp(-decline_rate * upper_age_bound) - exp(-decline_rate * lower_age_bound)
        )
        return 1 - unscaled_difference

    def age_precision_bounds(self, age: float, precision: str):
        """
        Get the precision factor based on the datetime precision.

        :param precision: The precision of the datetime ('year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond').
        :return: The corresponding precision factor.
        """
        precision_factors = {
            "year": 365.25,
            "month": 30,
            "day": 1,
            "hour": 1.0 / 24,
            "minute": 1.0 / (24 * 60),
            "second": 1.0 / (24 * 60 * 60),
            "microsecond": 1.0 / (24 * 60 * 60 * 1_000_000),
        }
        factor = precision_factors.get(precision, 1)
        lower_bound = floor(age / factor) * factor
        upper_bound = (floor(age / factor) + 1) * factor
        return lower_bound, upper_bound
