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
from metis.utils.logging import logger as main_logger
from metis.utils.logging import warn_unconfigured_columns
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

        if not ingestion_date_column or ingestion_date_column not in data.columns:
            self.logger.warning(
                f"Ingestion date column '{ingestion_date_column}' is not present in the data."
            )
            return results

        warn_unconfigured_columns(
            self.logger,
            set(data.columns),
            set(config.decline_rate_per_column.keys()),
            "decline rates",
        )

        ingestion_dates = pd.to_datetime(
            data[ingestion_date_column], **(config.to_datetime_kwargs or {})
        )
        ages_in_days = (
            (assessment_date - ingestion_dates).dt.total_seconds() / 60 / 60 / 24
        )
        precision_of_dates = (
            pd.Series(
                [config.simulated_timestamp_precision] * len(data), index=data.index
            )
            if config.simulated_timestamp_precision
            else data[ingestion_date_column].apply(determine_datetime_precision)
        )
        age_and_precision = pd.DataFrame(
            {"age": ages_in_days, "precision": precision_of_dates}
        )

        for col_name in data.columns:
            decline_rate = config.decline_rate_per_column.get(col_name)
            if decline_rate is None:
                continue

            timeliness = pd.Series(np.exp(-decline_rate * ages_in_days))
            certainty = age_and_precision.apply(
                lambda row: self.certainty(
                    row["age"],
                    decline_rate or 0,
                    row["precision"],
                ),
                axis=1,
            )
            for (index, timeliness_value), (_, certainty_value) in zip(
                timeliness.items(), certainty.items()
            ):
                result = DQResult(
                    mesTime=pd.Timestamp.now(),
                    DQvalue=timeliness_value,
                    DQdimension=DQDimension.TIMELINESS,
                    DQmetric=self.__class__.__name__,
                    columnNames=[col_name],
                    rowIndex=int(str(index)),
                    DQexplanation={
                        "certainty": certainty_value,
                    },
                    DQgranularity="cell",
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
