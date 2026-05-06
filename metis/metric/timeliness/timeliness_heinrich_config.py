import dataclasses
from dataclasses import dataclass
from typing import Dict

from metis.metric.config import MetricConfig
from metis.utils.datetime.datetime_precision import DTPrecision


@dataclass
class timeliness_heinrich_column_config:
    """
    Configuration class for a single column in the timeliness_heinrich metric (used as part of timeliness_heinrich_config).

    :param decline_rate: Decline rate for the column
    :param ingestion_date_column: Name of the column containing the ingestion date that should be used to calculate the age of the data for this column
    :param to_datetime_kwargs: Optional keyword arguments for pandas.to_datetime when parsing the date in ingestion_date_column.
    :param simulated_assessment_date: Optional simulated assessment date in string format. If not provided, the current date will be used. This can be used to simulate the assessment of data at a specific point in time, which is especially useful for testing and evaluation purposes.
    :param simulated_timestamp_precision: Optional simulated precision of each the timestamps in ingestion_date_column. If not provided, the precision is detected automatically. The precision is used to assess the certainty of the timeliness measurements.
    """

    decline_rate: float
    ingestion_date_column: str
    to_datetime_kwargs: Dict | None = None
    simulated_assessment_date: str | None = None
    simulated_timestamp_precision: DTPrecision | None = None


@dataclass
class timeliness_heinrich_config(MetricConfig):
    """
    Configuration class for the timeliness_heinrich metric.

    :param timeliness_config_per_column: Configuration for each column in the timeliness_heinrich metric. Each column can have a different decline rate and ingestion date column, which allows for a more fine-grained and accurate assessment of timeliness based on the specific characteristics of each column.
    """

    timeliness_config_per_column: Dict[str, timeliness_heinrich_column_config]

    def to_json(self):
        return {
            "name": self.__class__.__name__,
            "timeliness_config_per_column": {
                col: dataclasses.asdict(config)
                for col, config in self.timeliness_config_per_column.items()
            },
        }
