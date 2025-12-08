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
