from typing import List

import pandas as pd

from metis.metric.accuracy.accuracy_dataRange_config import (
    accuracy_dataRange_config,
)
from metis.metric.config import MetricConfig
from metis.metric.metric import Metric
from metis.profiling import value_range
from metis.utils.dq_dimension import DQDimension
from metis.utils.dq_granularity import DQGranularity
from metis.utils.result import DQResult


class accuracy_dataRange(Metric):
    """Acc-I-7: per column, share of values inside a required interval.

    DQvalue ∈ [0, 1] where 1.0 = every value inside the interval. The
    interval comes from ``metric_config.intervals[col]`` if provided,
    otherwise from the cached ``value_range`` profiling task when
    ``fallback="profiling"``. Columns with no available interval are
    skipped with a warning.
    """

    _gui_requires_reference: bool = False
    _gui_config_required: bool = False
    _gui_callable_config: bool = False
    _gui_recommended_granularities: frozenset = frozenset({DQGranularity.COLUMN})
    _gui_description: str = (
        "Per column, share of values that fall inside a required interval. "
        "Interval is supplied via metric_config.intervals or derived from the "
        "data's observed min/max (data profiling)."
    )

    def assess(
        self,
        data: pd.DataFrame,
        reference: pd.DataFrame | None = None,
        metric_config: str | MetricConfig | None = None,
    ) -> List[DQResult]:
        config = self.load_config(metric_config or "", accuracy_dataRange_config)
        results: List[DQResult] = []
        intervals = config.intervals or {}

        for col in data.columns:
            series = data[col]
            if not pd.api.types.is_numeric_dtype(series):
                self.logger.warning(
                    "accuracy_dataRange: skipping non-numeric column '%s'.", col
                )
                continue

            clean = series.dropna()
            if clean.empty:
                self.logger.warning(
                    "accuracy_dataRange: skipping column '%s' — no non-null values.", col
                )
                continue

            if col in intervals:
                lo, hi = intervals[col]
                source = "user"
            elif config.fallback == "profiling":
                bounds = value_range(series)
                if bounds["min"] is None:
                    self.logger.warning(
                        "accuracy_dataRange: skipping '%s' — profiling returned no bounds.",
                        col,
                    )
                    continue
                lo, hi = bounds["min"], bounds["max"]
                source = "profiling"
            else:
                self.logger.warning(
                    "accuracy_dataRange: skipping '%s' — no interval and fallback='skip'.",
                    col,
                )
                continue

            in_range = clean.between(lo, hi, inclusive="both")
            dq_value = float(in_range.mean())

            results.append(DQResult(
                timestamp=pd.Timestamp.now(),
                DQdimension=DQDimension.ACCURACY,
                DQmetric=self.__class__.__name__,
                DQgranularity=DQGranularity.COLUMN,
                DQvalue=dq_value,
                columnNames=[col],
                DQexplanation={
                    "interval": [float(lo), float(hi)],
                    "interval_source": source,
                    "in_range_count": int(in_range.sum()),
                    "out_of_range_count": int((~in_range).sum()),
                    "considered_count": int(len(clean)),
                },
            ))

        return results
