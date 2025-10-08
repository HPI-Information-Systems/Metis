from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from metis.metric.config import MetricConfig


@dataclass
class RuleConsistencyConfig(MetricConfig):
    """
    Configuration class for the RuleConsistency metric.
    """

    rules: Dict[
        str, List[Callable[[Any], float]]
    ]  # Dictionary of functions that define consistency rules for each column given by the key
