from typing import List

import pandas as pd

from metis.metric.accuracy.accuracy_outlierRisk_config import (
    accuracy_outlierRisk_config,
)
from metis.metric.config import MetricConfig
from metis.metric.metric import Metric
from metis.profiling import iqr_bounds
from metis.utils.data_profiling.single_column.value_distribution.outliers import (
    detect_outliers,
)
from metis.utils.dq_dimension import DQDimension
from metis.utils.dq_granularity import DQGranularity
from metis.utils.result import DQResult


class accuracy_outlierRisk(Metric):
    """Acc-I-4: per column, share of values *not* flagged as outliers.

    ISO/IEC 25024 defines this metric as ``outliers / total`` (lower is
    better). Metis inverts it so all accuracy metrics share the convention
    "1.0 = perfect"; the stored ``DQvalue`` is therefore
    ``1 - (outliers / total)``. ``DQexplanation["outlier_count"]`` and
    ``DQexplanation["considered_count"]`` are sufficient to recover the raw
    ISO ratio if needed.
    """

    _gui_requires_reference: bool = False
    _gui_config_required: bool = False
    _gui_callable_config: bool = False
    _gui_recommended_granularities: frozenset = frozenset({DQGranularity.COLUMN})
    _gui_description: str = (
        "Per column, share of values *not* flagged as outliers by a configurable "
        "detection method (default: IQR / Tukey). ISO/IEC 25024 defines Acc-I-4 "
        "as outliers / total (lower=better); Metis stores the inverted value "
        "1 - (outliers / total) so higher is always better and the metric "
        "aggregates with other accuracy metrics. See `outlier_count` and "
        "`considered_count` in DQexplanation to recover the raw ISO ratio."
    )

    def assess(
        self,
        data: pd.DataFrame,
        reference: pd.DataFrame | None = None,
        metric_config: str | MetricConfig | None = None,
    ) -> List[DQResult]:
        config = self.load_config(metric_config or "", accuracy_outlierRisk_config)
        params = config.method_params or {}
        results: List[DQResult] = []

        for col in data.columns:
            series = data[col]
            if not pd.api.types.is_numeric_dtype(series):
                self.logger.warning(
                    "accuracy_outlierRisk: skipping non-numeric column '%s'.", col
                )
                continue

            clean = series.dropna()
            if clean.empty:
                self.logger.warning(
                    "accuracy_outlierRisk: skipping column '%s'. No non-null values.", col
                )
                continue

            mask = detect_outliers(series, method=config.method, **params)
            # mask aligns with `series`; restrict to non-null positions
            mask_clean = mask.loc[clean.index]
            n = int(len(clean))
            n_outliers = int(mask_clean.sum())
            raw_inaccuracy = n_outliers / n
            dq_value = 1.0 - raw_inaccuracy  # <-- INVERSION to make "higher is better"

            explanation = {
                "method": config.method,
                "method_params": params,
                "outlier_count": n_outliers,
                "considered_count": n,
            }
            if config.method == "iqr":
                bounds = iqr_bounds(series, **params)
                explanation.update(bounds)

            results.append(DQResult(
                timestamp=pd.Timestamp.now(),
                DQdimension=DQDimension.ACCURACY,
                DQmetric=self.__class__.__name__,
                DQgranularity=DQGranularity.COLUMN,
                DQvalue=dq_value,
                columnNames=[col],
                DQexplanation=explanation,
            ))

        return results
