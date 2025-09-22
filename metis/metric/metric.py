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
    def assess(self, 
               data: pd.DataFrame, 
               reference: Union[pd.DataFrame, None] = None, 
               metric_config: Union[str, None] = None) -> List[DQResult]:
                """Assess data using this metric and return the results.

                Parameters
                - data: pd.DataFrame
                        The DataFrame that should be assessed by this metric. This is
                        the primary dataset under inspection.

                - reference: Optional[pd.DataFrame]
                        An optional, cleaned reference DataFrame that can act as a
                        clean version of the dataset. Metrics that need a canonical or
                        expected version of the data (for example correctness against a
                        known-good source) should accept and use this DataFrame. If not
                        needed by a metric, `None` is allowed.

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