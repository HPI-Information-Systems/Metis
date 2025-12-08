from dataclasses import dataclass
from typing import Dict

from metis.metric.config import MetricConfig


@dataclass
class TimelinessConfig(MetricConfig):
    """
    Configuration class for the TimelinessHeinrich metric.
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
