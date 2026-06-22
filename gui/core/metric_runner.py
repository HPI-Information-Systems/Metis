"""Run selected metrics directly via ``Metric().assess()`` with per-metric error isolation."""
from __future__ import annotations

import json
import os
import tempfile
import traceback
from dataclasses import dataclass
from typing import Callable

import pandas as pd

import metis.metric  # noqa: F401 — populates Metric.registry on import
from metis.metric.metric import Metric
from metis.utils.result import DQResult

from core.metric_catalog import get_catalog


@dataclass
class RunError:
    metric: str
    error: str
    tb: str


def run_all(
    metric_names: list[str],
    data: pd.DataFrame,
    reference: pd.DataFrame | None,
    configs: dict,
    on_progress: Callable[[int, int, str], None],
    max_rows_by_metric: dict[str, int] | None = None,
) -> tuple[list[DQResult], list[RunError]]:
    """
    Run each selected metric in sequence with isolated error handling.

    Errors in one metric do not stop subsequent ones. They are collected and
    returned alongside the successful results.

    :param metric_names: Ordered list of metric names to run.
    :param data: Input dataframe.
    :param reference: Optional reference dataframe (used by reference-based metrics).
    :param configs: Mapping of metric name to config object (or dict for the FD metric).
    :param on_progress: Callback ``on_progress(current_index, total, metric_name)``
        invoked before each metric run.
    :param max_rows_by_metric: Optional per-metric row cap (for cell-granularity
        metrics that produce very large result sets).
    :return: ``(results, errors)`` — the merged result list and any per-metric errors.
    """
    results: list[DQResult] = []
    errors: list[RunError] = []
    tmp_files: list[str] = []

    try:
        for i, name in enumerate(metric_names):
            on_progress(i, len(metric_names), name)
            try:
                metric_config = _prepare_config(name, configs.get(name), tmp_files)
                metric_data = data
                if max_rows_by_metric and name in max_rows_by_metric:
                    cap = max_rows_by_metric[name]
                    if cap > 0 and len(data) > cap:
                        metric_data = data.head(cap)
                batch = Metric.registry[name]().assess(metric_data, reference, metric_config)
                results.extend(batch)
            except Exception as e:
                errors.append(RunError(
                    metric=name,
                    error=str(e),
                    tb=traceback.format_exc(),
                ))
    finally:
        for path in tmp_files:
            try:
                os.unlink(path)
            except OSError:
                pass

    return results, errors


def _prepare_config(metric_name: str, config, tmp_files: list[str]):
    """
    Resolve the config payload that should be passed into ``Metric.assess``.

    For ``consistency_countFDViolations`` the GUI passes a dict, but the metric
    expects a JSON file path; this writes the dict to a temp file and returns
    the path. For metrics with a config dataclass and no user-provided config,
    the default config is instantiated so that ``Metric.load_config`` receives
    a valid object.

    :param metric_name: Metric the config belongs to.
    :param config: User-supplied config (may be ``None``).
    :param tmp_files: Mutable list collecting temp file paths for later cleanup.
    :return: The config payload to pass to ``Metric.assess``.
    """
    if metric_name == "consistency_countFDViolations" and isinstance(config, dict):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(config, tmp)
        tmp.close()
        tmp_files.append(tmp.name)
        return tmp.name

    if config is None:
        info = get_catalog().get(metric_name)
        if info and info.config_class is not None:
            return info.config_class()

    return config
