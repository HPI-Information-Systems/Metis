import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd

from metis.metric.config import MetricConfig


@dataclass(kw_only=True)
class consistency_ruleBasedPipino_config(MetricConfig):
    """
    Configuration class for the consistency_ruleBasedPipino metric.

    Accepts a dictionary mapping attribute names to lists of functions that define consistency rules.
    :param attribute_rules: Dictionary of functions that define consistency rules for each column given by the key
    :param tuple_rules: List of functions that define consistency rules for entire tuples
    """

    attribute_rules: Dict[str, List[Callable[[Any], bool]]] | None = None

    tuple_rules: List[Tuple[List[str], Callable[[pd.Series], bool]]] | None = None

    def to_json(self):
        return {
            "name": self.__class__.__name__,
            "attribute_rules": (
                {
                    column: [inspect.getsource(rule).strip() for rule in rules]
                    for column, rules in self.attribute_rules.items()
                }
                if self.attribute_rules
                else {}
            ),
            "tuple_rules": (
                [
                    (required_columns, inspect.getsource(rule).strip())
                    for required_columns, rule in self.tuple_rules
                ]
                if self.tuple_rules
                else []
            ),
        }
