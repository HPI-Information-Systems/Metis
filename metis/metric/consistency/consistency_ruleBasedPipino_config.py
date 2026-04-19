import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd

from metis.metric.config import MetricConfig


@dataclass(kw_only=True)
class consistency_ruleBasedPipino_config(MetricConfig):
    """
    Configuration class for the consistency_ruleBasedPipino metric.

    Accepts a dictionary mapping column names to lists of functions that define consistency rules.
    :param column_rules: Dictionary of functions that define consistency rules for each column given by the key
    :param tuple_rules: List of functions that define consistency rules for entire tuples
    :param skip_null_values: Whether to skip null values when assessing consistency. If True, null values will be ignored. For tuple rules, a tuple is skipped if all of the values in the tuple are null.
    """

    column_rules: Dict[str, List[Callable[[Any], bool]]] | None = None
    tuple_rules: List[Callable[[pd.Series], bool]] | None = None
    skip_null_values: bool = False

    def to_json(self):
        return {
            "name": self.__class__.__name__,
            "column_rules": (
                {
                    column: [inspect.getsource(rule).strip() for rule in rules]
                    for column, rules in self.column_rules.items()
                }
                if self.column_rules
                else {}
            ),
            "tuple_rules": (
                [inspect.getsource(rule).strip() for rule in self.tuple_rules]
                if self.tuple_rules
                else []
            ),
            "skip_null_values": self.skip_null_values,
        }
