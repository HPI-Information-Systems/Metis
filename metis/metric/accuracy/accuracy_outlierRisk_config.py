from dataclasses import dataclass
from typing import Any, Dict

from metis.metric.config import MetricConfig
from metis.utils.data_profiling.single_column.value_distribution.outliers import (
    available_methods,
)


@dataclass
class accuracy_outlierRisk_config(MetricConfig):
    """Configuration for ``accuracy_outlierRisk`` (ISO/IEC 25024 Acc-I-4).

    :param method: Name of an outlier-detection strategy registered in
        :mod:`metis.utils.data_profiling.single_column.value_distribution.outliers`.
        Default ``"iqr"`` (Tukey method, ``1.5 * IQR``).
    :param method_params: Extra keyword arguments forwarded to the strategy.
        For ``"iqr"`` this is e.g. ``{"multiplier": 1.5}``.
    """

    method: str = "iqr"
    method_params: Dict[str, Any] | None = None

    def to_json(self):
        return {
            "name": self.__class__.__name__,
            "method": self.method,
            "method_params": self.method_params or {},
        }

    def validate(self):
        methods = available_methods()
        if self.method not in methods:
            raise ValueError(
                f"method must be one of {methods} but was '{self.method}'."
            )
        if self.method_params is not None and not isinstance(self.method_params, dict):
            raise ValueError(
                f"method_params must be a dict or None, got {type(self.method_params)}."
            )
