from dataclasses import dataclass
from typing import Dict

from metis.metric.config import MetricConfig


@dataclass
class CurrencyConfig(MetricConfig):
    """
    Configuration class for the Currency metric.
    """

    decline_rate_per_column: Dict[str, float]
    ingestion_date_column: str
    simulated_assessment_date: str | None = None
