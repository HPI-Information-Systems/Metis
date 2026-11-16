from typing import Iterable, List, Optional

import pandas as pd

from metis.metric.accuracy._strategies.domain_membership import DOMAIN_STRATEGIES
from metis.metric.accuracy.accuracy_syntacticDomain_config import (
    accuracy_syntacticDomain_config,
)
from metis.metric.config import MetricConfig
from metis.metric.metric import Metric
from metis.metric.metric_meta import MetricMeta
from metis.utils.dq_dimension import DQDimension
from metis.utils.dq_granularity import DQGranularity
from metis.utils.reference_loader import load_reference_frame
from metis.utils.result import DQResult


class accuracy_syntacticDomain(Metric):
    """Acc-I-1: per column, share of values that belong to a domain D.

    The domain for a column is resolved (in order) from:
      1. ``metric_config.domains[col]`` (explicit list/set)
      2. The ``reference`` DataFrame: same-named column, or its single column
         when reference has only one.
      3. The strategy's own source. Only ``method="wordnet"`` has one.
    """

    meta = MetricMeta(
        label="Syntactic Domain",
        description=(
            "Per column, share of values that belong to a configurable reference "
            "domain. Domain can be a per-column list (metric_config.domains), a "
            "reference DataFrame, or NLTK WordNet."
        ),
        dimension=DQDimension.ACCURACY,
        granularities=frozenset({DQGranularity.COLUMN}),
        standard="ISO/IEC 25024:2015 Acc-I-1",
    )

    def assess(
        self,
        data: pd.DataFrame,
        *,
        metric_config: str | MetricConfig | None = None,
    ) -> List[DQResult]:
        config = self.load_config(metric_config or "", accuracy_syntacticDomain_config)
        reference = load_reference_frame(config.reference)
        strategy = DOMAIN_STRATEGIES[config.method]
        params = config.method_params or {}
        results: List[DQResult] = []

        for col in data.columns:
            series = data[col].dropna()
            if series.empty:
                self.logger.warning(
                    "accuracy_syntacticDomain: skipping '%s'. No non-null values.", col
                )
                continue

            domain, source = self._resolve_domain(col, config, reference)
            if config.method == "exact_match" and domain is None:
                self.logger.warning(
                    "accuracy_syntacticDomain: skipping '%s'. No domain available.", col
                )
                continue

            in_domain = strategy(series, domain, **params)
            dq_value = float(in_domain.mean())

            results.append(DQResult(
                timestamp=pd.Timestamp.now(),
                DQdimension=DQDimension.ACCURACY,
                DQmetric=self.__class__.__name__,
                DQgranularity=DQGranularity.COLUMN,
                DQvalue=dq_value,
                columnNames=[col],
                DQexplanation={
                    "method": config.method,
                    "domain_source": source,
                    "domain_size": (len(domain) if domain is not None else None),
                    "in_domain_count": int(in_domain.sum()),
                    "considered_count": int(len(series)),
                },
            ))

        return results

    def _resolve_domain(
        self,
        col: str,
        config: accuracy_syntacticDomain_config,
        reference: Optional[pd.DataFrame],
    ) -> tuple[Optional[Iterable], str]:
        if config.domains and col in config.domains:
            return list(config.domains[col]), "config"
        if isinstance(reference, pd.DataFrame):
            if col in reference.columns:
                return reference[col].dropna().unique().tolist(), "reference"
            if reference.shape[1] == 1:
                return reference.iloc[:, 0].dropna().unique().tolist(), "reference"
        if config.method == "wordnet":
            return None, "wordnet"   # strategy has its own internal source
        return None, "none"
