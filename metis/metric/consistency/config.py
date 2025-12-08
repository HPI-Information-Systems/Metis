import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from metis.metric.config import MetricConfig


@dataclass
class ConsistencyConfig(MetricConfig):
    """
    Configuration class for the Consistency metric.
    """

    rules: Dict[
        str, List[Callable[[Any], float]]
    ]  # Dictionary of functions that define consistency rules for each column given by the key


@dataclass
class ConsistencyRuleBasedHinrichsConfig(MetricConfig):
    """
    Configuration class for the RuleBasedHinrichs metric.
    """

    rules: Dict[
        str, List[Callable[[Any], float]]
    ]  # Dictionary of functions that define consistency rules for each column given by the key

    def to_json(self):
        return {
            "name": self.__class__.__name__,
            "rules": {
                column: [inspect.getsource(rule).strip() for rule in rules]
                for column, rules in self.rules.items()
            },
        }
