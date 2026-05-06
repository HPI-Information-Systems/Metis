"""DQResult ↔ JSON-safe dict conversion."""
from __future__ import annotations

import math

import pandas as pd

from metis.utils.dq_dimension import DQDimension
from metis.utils.dq_granularity import DQGranularity
from metis.utils.result import DQResult


def result_to_dict(r: DQResult) -> dict:
    """
    Convert a :class:`DQResult` to a JSON-serializable dict.

    :param r: The result object to serialize.
    :return: A dict with all DQResult fields in canonical form.
    """
    value = r.DQvalue
    if value is not None and isinstance(value, float) and math.isnan(value):
        value = None
    return {
        "timestamp": r.timestamp.isoformat() if r.timestamp is not None else None,
        "DQdimension": str(r.DQdimension),
        "DQmetric": r.DQmetric,
        "DQgranularity": str(r.DQgranularity),
        "DQvalue": value,
        "DQexplanation": r.DQexplanation,
        "runtime": r.runtime,
        "tableName": r.tableName,
        "columnNames": r.columnNames,
        "rowIndex": r.rowIndex,
        "experimentTag": r.experimentTag,
        "dataset": r.dataset,
        "configJson": r.configJson,
    }


def dict_to_result(d: dict) -> DQResult:
    """
    Reconstruct a :class:`DQResult` from a serialized dict.

    :param d: A dict produced by :func:`result_to_dict` (or a compatible payload).
    :return: A new ``DQResult`` instance.
    """
    return DQResult(
        timestamp=pd.Timestamp(d["timestamp"]) if d.get("timestamp") else pd.Timestamp.now(),
        DQdimension=DQDimension(d["DQdimension"]),
        DQmetric=d["DQmetric"],
        DQgranularity=DQGranularity(d["DQgranularity"]),
        DQvalue=d["DQvalue"],
        DQexplanation=d.get("DQexplanation"),
        runtime=d.get("runtime"),
        tableName=d.get("tableName"),
        columnNames=d.get("columnNames"),
        rowIndex=d.get("rowIndex"),
        experimentTag=d.get("experimentTag"),
        dataset=d.get("dataset"),
        configJson=d.get("configJson"),
    )
