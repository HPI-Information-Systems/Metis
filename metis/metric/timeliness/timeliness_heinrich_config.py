from dataclasses import dataclass
from typing import Dict

from metis.metric.config import MetricConfig


@dataclass
class timeliness_heinrich_config(MetricConfig):
    """
    Configuration class for the timeliness_heinrich metric.

    :param decline_rate_per_column: Decline rate specific to each column
    :param ingestion_date_column: Name of the column containing the ingestion date of each tuple
    :param simulated_assessment_date: Optional simulated assessment date in string format. If not provided, the current date will be used.
    """

    decline_rate_per_column: Dict[str, float]
    ingestion_date_column: str
    simulated_assessment_date: str | None = None

    def to_json(self):
        return {
            "name": self.__class__.__name__,
            "decline_rate_per_column": self.decline_rate_per_column,
            "ingestion_date_column": self.ingestion_date_column,
            "simulated_assessment_date": self.simulated_assessment_date,
        }
