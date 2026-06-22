from typing import List

import pandas as pd

from metis.metric.accuracy.accuracy_semanticReference_config import (
    accuracy_semanticReference_config,
)
from metis.metric.config import MetricConfig
from metis.metric.metric import Metric
from metis.utils.dq_dimension import DQDimension
from metis.utils.dq_granularity import DQGranularity
from metis.utils.result import DQResult


class accuracy_semanticReference(Metric):
    """Acc-I-2: per column, share of values that match a gold standard.

    Comparison is exact (``==``); NaN matches NaN. Use ``correctness_heinrich``
    for fuzzy / numeric-tolerance comparison. Reference is required.

    A gold standard is rarely available for the full table. 
    If ``reference`` covers a sample, invoke this metric on ``data.loc[reference.index]``
    (or align via ``key_column``) so denominators reflect the sample, not the full table.
    """

    _gui_requires_reference: bool = True
    _gui_config_required: bool = False
    _gui_callable_config: bool = False
    _gui_recommended_granularities: frozenset = frozenset({DQGranularity.COLUMN})
    _gui_description: str = (
        "Per column, share of values that exactly match the corresponding value "
        "in a gold-standard reference DataFrame. Alignment is positional by "
        "default, or by metric_config.key_column."
    )

    def assess(
        self,
        data: pd.DataFrame,
        reference: pd.DataFrame | None = None,
        metric_config: str | MetricConfig | None = None,
    ) -> List[DQResult]:
        if reference is None:
            raise ValueError(
                "accuracy_semanticReference requires a reference DataFrame."
            )

        config = self.load_config(metric_config or "", accuracy_semanticReference_config)

        if config.key_column is not None:
            if config.key_column not in data.columns:
                raise ValueError(
                    f"key_column '{config.key_column}' not in data.columns."
                )
            if config.key_column not in reference.columns:
                raise ValueError(
                    f"key_column '{config.key_column}' not in reference.columns."
                )
            aligned_data = data.set_index(config.key_column)
            aligned_ref = reference.set_index(config.key_column)
            common = aligned_data.index.intersection(aligned_ref.index)
            if len(common) == 0:
                self.logger.warning(
                    "accuracy_semanticReference: no overlapping keys; emitting no results."
                )
                return []
            aligned_data = aligned_data.loc[common]
            aligned_ref = aligned_ref.loc[common]
            alignment = "key_column"
        else:
            if len(data) != len(reference):
                raise ValueError(
                    f"data ({len(data)}) and reference ({len(reference)}) must have "
                    f"the same length when key_column is None."
                )
            aligned_data = data
            aligned_ref = reference
            alignment = "row_index"

        results: List[DQResult] = []
        for col in aligned_data.columns:
            if col not in aligned_ref.columns:
                self.logger.warning(
                    "accuracy_semanticReference: skipping '%s'. Column missing in reference.",
                    col,
                )
                continue

            d = aligned_data[col].reset_index(drop=True)
            r = aligned_ref[col].reset_index(drop=True)
            n = len(d)
            if n == 0:
                self.logger.warning(
                    "accuracy_semanticReference: skipping '%s'. Empty after alignment.", col
                )
                continue

            both_nan = d.isna() & r.isna()
            match = (d == r) | both_nan
            dq_value = float(match.sum() / n)

            results.append(DQResult(
                timestamp=pd.Timestamp.now(),
                DQdimension=DQDimension.ACCURACY,
                DQmetric=self.__class__.__name__,
                DQgranularity=DQGranularity.COLUMN,
                DQvalue=dq_value,
                columnNames=[col],
                DQexplanation={
                    "considered_count": int(n),
                    "match_count": int(match.sum()),
                    "mismatch_count": int((~match).sum()),
                    "key_column": config.key_column,
                    "alignment": alignment,
                },
            ))

        return results
