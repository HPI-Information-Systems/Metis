from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Union

from metis.utils.result import DQResult

class Metric(ABC):
    """
    Abstract base class for metrics.
    All metric classes should inherit from this class and implement the `compute` method.
    """
    registry = {}

    def __init_subclass__(cls):
        super().__init_subclass__()
        Metric.registry[cls.__name__] = cls

    @abstractmethod
    def assess(self, data: pd.DataFrame, reference: Union[pd.DataFrame, None] = None, metric_config: Union[str, None] = None) -> List[DQResult]:
        pass