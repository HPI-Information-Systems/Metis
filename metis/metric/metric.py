from abc import ABC, abstractmethod
from typing import Any, List, TypeVar

import pandas as pd

from metis.metric.config import MetricConfig
from metis.utils.json_loading import load_json_string_or_path
from metis.utils.logging import logger as main_logger
from metis.utils.result import DQResult

CONF = TypeVar("CONF", bound=MetricConfig)


class Metric(ABC):
    """
    Abstract base class for metrics.
    All metric classes should inherit from this class and implement the `assess` method.
    """

    registry = {}

    def __init_subclass__(cls):
        super().__init_subclass__()
        Metric.registry[cls.__name__] = cls

    def __init__(self) -> None:
        self.logger = main_logger.getChild(self.__class__.__name__)

    @abstractmethod
    def assess(
        self,
        data: pd.DataFrame,
        metric_config: str | MetricConfig | None = None,
    ) -> List[DQResult]:
        """Assess data using this metric and return the results.

        Parameters
        - data: pd.DataFrame
                The DataFrame that should be assessed by this metric. This is
                the primary dataset under inspection.

        - metric_config: Optional[str]
                Optional path or JSON string containing metric-specific
                configuration. Use this to keep the method signature compact;
                all metric-specific parameters (thresholds, aggregation options,
                etc.) can be stored here.

        Returns
        - List[DQResult]
                A list of `DQResult` objects. Each `DQResult` instance captures
                one assessed value produced by the metric. For metrics that
                operate at the column level, there should be one `DQResult` per
                column; for table-level metrics typically a single `DQResult`
                is returned. Implementations are free to return multiple
                results for any logical decomposition the metric provides
                (e.g., per-column, per-partition, per-check).

        Notes
        - Implementations must avoid mutating the
            input `data` and `reference` DataFrames in-place.
        - `metric_config` should be parsed by the implementation and any
            invalid config should raise a clear exception describing the
            expected format.

        Examples
        - Column-level completeness metric: returns one `DQResult` per
            column with the fraction of non-null values.
        - Correctness metric against a reference: compares `data` to
            `reference` and returns one `DQResult` per cell in the input table containing the
            agreement score.
        """
        raise NotImplementedError()

    def load_config(self, config: Any, model: type[CONF]) -> CONF:
        """
        Load metric-specific configuration from a JSON file path, JSON string or the correct config model instance. Also validates the configuration using its validate method.

        :param config: Path to the JSON configuration file, a JSON string or an instance of the config model.
        :return: An instance of the metric-specific configuration class.
        """
        if isinstance(config, model):
            config.validate()
            return config

        if isinstance(config, str):
            parsed_config = model.from_dict(load_json_string_or_path(config))
            parsed_config.validate()
            return parsed_config

        raise TypeError(
            f"Invalid config type: {type(config)}. Expected str or {model}."
        )
