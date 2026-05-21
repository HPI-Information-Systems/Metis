from typing import List, Union

import pandas as pd

from metis.metric.config import MetricConfig
from metis.metric.consistency.consistency_cpfd_config import (
    consistency_cpfd_config,
)
from metis.metric.metric import Metric
from metis.profiling.data_profile_manager import DataProfileManager
from metis.utils.dq_dimension import DQDimension
from metis.utils.dq_granularity import DQGranularity
from metis.utils.result import DQResult


class consistency_cpfd(Metric):
    """Automated consistency assessment via partial functional dependencies.

    Implements the cpfd metric from Seeger et al. (VLDB QDB'26):

        cpfd(F) = sum_{X->A in F} omega(X,A) * r(X,A) / sum_{X->A in F} omega(X,A)

    where omega is the gpdep-based genuineness weight in [0, 1] and r is 1
    iff the partial threshold rho equals 1 (i.e., the FD holds exactly).

    The pFDs together with their rho and gpdep weights are expected to be
    pre-computed (e.g., by the Java HyFD extension from cpfd-reproducibility)
    and loaded into the DataProfileManager via the "pfd" importer.
    """

    def assess(
        self,
        data: pd.DataFrame,
        reference: Union[pd.DataFrame, None] = None,
        metric_config: Union[MetricConfig, str, None] = None,
    ) -> List[DQResult]:
        config = self.load_config(metric_config or "{}", consistency_cpfd_config)

        manager = DataProfileManager.get_instance()
        pfds = manager._query_by_task(
            config.pfd_task_name,
            dataset=manager.dataset,
            table=manager.table,
        )

        if not pfds:
            self.logger.warning(
                "No partial FDs found under task '%s' for dataset=%s table=%s. "
                "Import pFDs via the 'pfd' data-profile importer before running this metric.",
                config.pfd_task_name,
                manager.dataset,
                manager.table,
            )
            return []

        total_weight = sum(float(p.get("gpdep", 0.0)) for p in pfds)
        if total_weight == 0.0:
            score = 0.0
        else:
            score = sum(
                float(p.get("gpdep", 0.0))
                * (1.0 if float(p.get("partial", 0.0)) >= 1.0 else 0.0)
                for p in pfds
            ) / total_weight

        breakdown = [
            {
                "lhs": p["lhs"],
                "rhs": p["rhs"],
                "partial": float(p.get("partial", 0.0)),
                "gpdep": float(p.get("gpdep", 0.0)),
                "r": 1 if float(p.get("partial", 0.0)) >= 1.0 else 0,
            }
            for p in pfds
        ]

        column_names = sorted({c for p in pfds for c in p["lhs"] + [p["rhs"]]})

        return [
            DQResult(
                timestamp=pd.Timestamp.now(),
                DQdimension=DQDimension.CONSISTENCY,
                DQmetric=self.__class__.__name__,
                DQgranularity=DQGranularity.TABLE,
                DQvalue=score,
                DQexplanation={
                    "num_pfds": len(pfds),
                    "total_weight": total_weight,
                    "pfds": breakdown,
                },
                columnNames=column_names,
                configJson=config.to_json(),
            )
        ]
