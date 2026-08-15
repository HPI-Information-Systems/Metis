"""Per-metric configuration needed to run a metric in the sweep.

Several metrics refuse to run without a config, and two need reference data.
Without these fixtures the sweep would silently cover only part of the
catalogue, which is worse than not running it at all.
"""
from __future__ import annotations

import importlib
import json
import pathlib
from typing import Any

import pandas as pd

import metis.metric  # noqa: F401 - populates Metric.registry
from metis.metric.metric import Metric
from metis.metric.accuracy.accuracy_dataRange_config import accuracy_dataRange_config
from metis.metric.accuracy.accuracy_semanticReference_config import (
    accuracy_semanticReference_config,
)
from metis.metric.accuracy.accuracy_syntacticDomain_config import (
    accuracy_syntacticDomain_config,
)
from metis.metric.consistency.consistency_ruleBasedHinrichs_config import (
    consistency_ruleBasedHinrichs_config,
)
from metis.metric.consistency.consistency_ruleBasedPipino_config import (
    consistency_ruleBasedPipino_config,
)
from metis.metric.correctness.correctness_heinrich_config import (
    correctness_heinrich_config,
)
from metis.metric.minimality.minimality_clustering_config import (
    minimality_clustering_config,
)
from metis.metric.timeliness.timeliness_heinrich_config import (
    timeliness_heinrich_column_config,
    timeliness_heinrich_config,
)

from benchmarks.datagen import _CATEGORIES

SKIPPED: dict[str, str] = {
    "readability_llm": (
        "Loads a local HuggingFace language model (default "
        "Qwen/Qwen2.5-3B-Instruct), so its runtime measures model inference "
        "rather than Metis. Needs the optional transformers dependency."
    ),
    "completeness_nullAndDMVRatio": (
        "Requires the FAHES native library, which is not built in every "
        "environment. See _NATIVE_LIB_CHECKS in gui/core/metric_catalog.py."
    ),
}


def config_for(metric_name: str, frame: pd.DataFrame, workdir: pathlib.Path) -> Any:
    """
    Build the config a metric needs to run against a generated frame.

    :param metric_name: Registry name of the metric.
    :param frame: The frame the metric will be run on.
    :param workdir: Directory for any file the config has to point at.
    :return: A config object, a file path, or ``None`` when the metric needs
        no configuration.
    """
    if metric_name == "consistency_countFDViolations":
        return _fd_config_path(frame, workdir)

    if metric_name in ("consistency_ruleBasedHinrichs", "consistency_ruleBasedPipino"):
        return _rule_config(metric_name, frame)

    if metric_name == "timeliness_heinrich":
        return _timeliness_config(frame)

    if metric_name == "accuracy_dataRange":
        return _data_range_config(frame)

    if metric_name == "accuracy_syntacticDomain":
        return _syntactic_domain_config(frame)

    if metric_name == "minimality_clustering":
        return minimality_clustering_config(use_semhash=False, similarity_threshold=0.85)

    if metric_name == "accuracy_semanticReference":
        return accuracy_semanticReference_config(reference=frame.copy())

    if metric_name == "correctness_heinrich":
        return correctness_heinrich_config(reference=frame.copy())

    return _default_config(metric_name)


def _default_config(metric_name: str):
    """
    Instantiate a metric's config class by naming convention, if it has one.

    Returning a bare ``None`` is not safe. ``Metric.load_config`` rejects
    ``None`` outright, and the completeness metrics pass ``metric_config``
    straight through without the ``metric_config or ""`` idiom the accuracy
    metrics use, so they raise ``TypeError`` on a ``None`` config. The sweep
    would then record them as errors rather than measuring them. This mirrors
    what ``_prepare_config`` in ``gui/core/metric_runner.py`` already does for
    the GUI.

    :param metric_name: Registry name of the metric.
    :return: A default-constructed config, or ``None`` if the metric has no
        config class or its config cannot be built without arguments.
    """
    metric_cls = Metric.registry[metric_name]
    try:
        mod = importlib.import_module(f"{metric_cls.__module__}_config")
    except ImportError:
        return None
    config_class = getattr(mod, f"{metric_cls.__name__}_config", None)
    if config_class is None:
        return None
    try:
        return config_class()
    except TypeError:
        # Config has required fields; the explicit branches above cover those.
        return None


def _first_column_of_kind(frame: pd.DataFrame, prefix: str) -> str | None:
    """Return the first generated column whose name starts with ``prefix``."""
    for col in frame.columns:
        if str(col).startswith(prefix):
            return str(col)
    return None


def _columns_of_kind(frame: pd.DataFrame, prefix: str) -> list[str]:
    """Return every generated column whose name starts with ``prefix``."""
    return [str(col) for col in frame.columns if str(col).startswith(prefix)]


def _fd_config_path(frame: pd.DataFrame, workdir: pathlib.Path) -> str:
    """Write a functional dependency spec over two generated columns."""
    determinant = _first_column_of_kind(frame, "categorical") or str(frame.columns[0])
    dependent = _first_column_of_kind(frame, "text") or str(frame.columns[-1])
    path = workdir / "fd_config.json"
    path.write_text(json.dumps({determinant: [dependent]}), encoding="utf-8")
    return str(path)


def _rule_config(metric_name: str, frame: pd.DataFrame):
    """Build a one-rule config for the two rule-based consistency metrics."""
    column = _first_column_of_kind(frame, "integer") or str(frame.columns[0])

    if metric_name == "consistency_ruleBasedPipino":
        return consistency_ruleBasedPipino_config(
            attribute_rules={column: [lambda v: v is not None]},
            tuple_rules=None,
        )

    return consistency_ruleBasedHinrichs_config(
        attribute_rules={column: [lambda v: 0.0 if v is not None else 1.0]},
        tuple_rules=None,
    )


def _timeliness_config(frame: pd.DataFrame):
    """Point the timeliness metric at the first generated date column."""
    column = _first_column_of_kind(frame, "date")
    if column is None:
        raise ValueError(
            "timeliness_heinrich needs a date column. Generate at least 5 "
            "columns so the date kind appears."
        )
    return timeliness_heinrich_config(
        timeliness_config_per_column={
            column: timeliness_heinrich_column_config(
                decline_rate=0.2,
                ingestion_date_column=column,
                # The generated date column holds real Timestamp values (not
                # strings), so the metric's automatic precision detection
                # (dateutil.parser.parse) cannot run on it. The generator
                # only varies whole-day offsets, so "day" is accurate here.
                simulated_timestamp_precision="day",
            )
        }
    )


def _data_range_config(frame: pd.DataFrame) -> accuracy_dataRange_config:
    """
    Give every generated numeric column an interval so the metric actually
    checks cells instead of skipping them for want of a range.

    The default config (``intervals=None``) skips every column: numeric
    columns for lack of an interval, non-numeric columns by type. Without an
    explicit interval per numeric column the metric would time the cost of
    skipping rather than the cost of the range check itself.

    :param frame: The frame the metric will be run on.
    :return: A config with an interval for every ``integer_*``/``float_*``
        column, wide enough to cover the generator's ranges.
    """
    intervals: dict[str, tuple[float, float]] = {}
    for column in _columns_of_kind(frame, "integer"):
        intervals[column] = (0, 1_000_000)
    for column in _columns_of_kind(frame, "float"):
        intervals[column] = (0.0, 200.0)
    return accuracy_dataRange_config(intervals=intervals, fallback="skip")


def _syntactic_domain_config(frame: pd.DataFrame) -> accuracy_syntacticDomain_config:
    """
    Give every generated categorical column a domain so the metric actually
    checks cells instead of reporting "no domain available".

    :param frame: The frame the metric will be run on.
    :return: A config with a domain for every ``categorical_*`` column, set
        to :data:`benchmarks.datagen._CATEGORIES` so it cannot drift from the
        values the generator actually produces.
    """
    domains = {
        column: list(_CATEGORIES)
        for column in _columns_of_kind(frame, "categorical")
    }
    return accuracy_syntacticDomain_config(domains=domains)
