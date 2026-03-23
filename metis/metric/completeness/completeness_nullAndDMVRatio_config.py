from dataclasses import dataclass
from typing import Literal

from metis.metric.config import MetricConfig

VALID_AGGREGATION_AXES = ["index", "columns", None]


@dataclass
class completeness_nullAndDMVRatio_config(MetricConfig):
    """
    Configuration class for the completeness_nullAndDMVRatio metric.

    :param aggregation_axis: Axis along which to aggregate completeness ('index': aggregate each column; 'columns': aggregate each row, None (default): no aggregation).
    :param aggregate_all: Whether to aggregate all completeness results into a single value for the whole input data.
    """

    aggregation_axis: Literal["index", "columns", None] = None
    aggregate_all: bool = False

    def to_json(self):
        return {
            "name": self.__class__.__name__,
            "aggregation_axis": self.aggregation_axis,
            "aggregate_all": self.aggregate_all,
        }

    def validate(self):
        if self.aggregation_axis not in VALID_AGGREGATION_AXES:
            raise ValueError(
                f"aggregation_axis must be one of {VALID_AGGREGATION_AXES} but was {self.aggregation_axis}"
            )
        if not isinstance(self.aggregate_all, bool):
            raise ValueError(
                f"aggregate_all must be a boolean value but was {type(self.aggregate_all)}"
            )
