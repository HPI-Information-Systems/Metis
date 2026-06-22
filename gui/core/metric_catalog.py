"""Build a catalog of :class:`MetricInfo` objects from the Metis metric registry."""
from __future__ import annotations

import dataclasses
import importlib
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import metis.metric  # noqa: F401 — triggers __init_subclass__ for all metrics
from metis.metric.metric import Metric
from metis.utils.dq_granularity import DQGranularity


@dataclass
class ConfigField:
    name: str
    annotation: Any
    default: Any
    has_default: bool


@dataclass
class MetricInfo:
    name: str
    dimension: str
    config_class: type | None
    fd_json_config: bool
    requires_reference: bool
    config_required: bool
    callable_config: bool
    produces_levels: frozenset[DQGranularity]
    recommended_granularities: frozenset[DQGranularity]
    config_fields: list[ConfigField]
    description: str = ""
    unavailable_reason: str | None = None

    @property
    def cell_granularity(self) -> bool:
        return DQGranularity.CELL in self.produces_levels


_NATIVE_LIB_CHECKS: dict[str, tuple[Path, str]] = {
    "completeness_nullAndDMVRatio": (
        Path(__file__).parent.parent.parent
        / "metis/utils/disguised_missing_values/fahes/lib/FAHES_Code/libFahes.so",
        "Requires FAHES native library (libFahes.so). "
        "Clone https://github.com/qcri/FAHES_Code.git into "
        "metis/utils/disguised_missing_values/fahes/lib/FAHES_Code and run make.",
    ),
}

_catalog: dict[str, MetricInfo] | None = None


def get_catalog() -> dict[str, MetricInfo]:
    """
    Return the lazily-built metric catalog, keyed by metric name.

    :return: Mapping ``{metric_name: MetricInfo}`` covering every registered metric.
    """
    global _catalog
    if _catalog is not None:
        return _catalog

    _catalog = {}
    for name, cls in Metric.registry.items():
        dimension = name.split("_")[0].capitalize()
        fd_json_config = name == "consistency_countFDViolations"
        requires_reference = getattr(cls, "_gui_requires_reference", False)
        config_required = getattr(cls, "_gui_config_required", False)
        callable_config = getattr(cls, "_gui_callable_config", False)
        description = getattr(cls, "_gui_description", "")

        recommended = frozenset(getattr(cls, "_gui_recommended_granularities", frozenset()))
        is_cell = getattr(cls, "_gui_cell_granularity", False)
        produces_levels = recommended | (
            {DQGranularity.CELL} if is_cell and DQGranularity.CELL not in recommended else frozenset()
        )

        if fd_json_config:
            config_class = None
            config_fields = []
        else:
            config_class = _find_config_class(cls)
            config_fields = _extract_fields(config_class) if config_class else []

        unavailable_reason = None
        if name in _NATIVE_LIB_CHECKS:
            lib_path, reason = _NATIVE_LIB_CHECKS[name]
            if not lib_path.exists():
                unavailable_reason = reason

        _catalog[name] = MetricInfo(
            name=name,
            dimension=dimension,
            config_class=config_class,
            fd_json_config=fd_json_config,
            requires_reference=requires_reference,
            config_required=config_required,
            callable_config=callable_config,
            produces_levels=produces_levels,
            recommended_granularities=recommended,
            config_fields=config_fields,
            description=description,
            unavailable_reason=unavailable_reason,
        )

    return _catalog


def get_compute_blockers(selected: list[str], configs: dict) -> list[str]:
    """
    Return reasons why the selected metrics cannot be computed yet.

    :param selected: Metric names the user has selected.
    :param configs: Current per-metric config payloads.
    :return: A list of blocker messages (empty when ready to compute).
    """
    catalog = get_catalog()
    blockers = []
    for name in selected:
        info = catalog.get(name)
        if not info:
            continue
        if info.unavailable_reason:
            blockers.append(f"**{name}** is unavailable: {info.unavailable_reason}")
        elif info.config_required and not configs.get(name):
            blockers.append(f"**{name}** requires configuration — expand it in the Metrics step.")
    return blockers


def get_metrics_by_dimension() -> dict[str, list[MetricInfo]]:
    """
    Group metrics by dimension.

    :return: Mapping ``{dimension: [MetricInfo, ...]}`` sorted alphabetically by dimension.
    """
    catalog = get_catalog()
    by_dim: dict[str, list[MetricInfo]] = {}
    for info in catalog.values():
        by_dim.setdefault(info.dimension, []).append(info)
    return dict(sorted(by_dim.items()))


def _find_config_class(metric_cls: type) -> type | None:
    """
    Resolve the config class for a metric by convention: same package, ``{name}_config`` module.

    :param metric_cls: The metric class.
    :return: The config class, or ``None`` if no matching ``{name}_config`` module exists.
    """
    module_name = metric_cls.__module__
    config_module_name = f"{module_name}_config"
    try:
        mod = importlib.import_module(config_module_name)
        return getattr(mod, f"{metric_cls.__name__}_config", None)
    except ImportError:
        return None


def _extract_fields(config_class: type) -> list[ConfigField]:
    """
    Inspect a config dataclass and return its field metadata.

    :param config_class: A ``MetricConfig`` subclass.
    :return: A list of :class:`ConfigField` entries (empty on inspection failure).
    """
    try:
        fields = []
        hints = {}
        try:
            hints = typing.get_type_hints(config_class)
        except Exception:
            pass
        for f in dataclasses.fields(config_class):
            annotation = hints.get(f.name, f.type)
            has_default = not isinstance(f.default, dataclasses._MISSING_TYPE)
            default = f.default if has_default else None
            fields.append(ConfigField(
                name=f.name,
                annotation=annotation,
                default=default,
                has_default=has_default,
            ))
        return fields
    except Exception:
        return []
