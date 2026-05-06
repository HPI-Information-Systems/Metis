"""Stable metric → color mapping and short-name helpers for multi-metric charts.

Used by dimension-tab renderers so the same metric keeps one color across the
grouped bar chart, overlay histogram, trend strip and worst-rows table.
"""
from __future__ import annotations

from theme import HPI_PALETTE

_PALETTE: tuple[str, ...] = HPI_PALETTE


def metric_colors(metric_names: list[str]) -> dict[str, str]:
    """
    Return a stable mapping from metric name to hex color.

    Sorting the metric names first makes the color assignment deterministic
    regardless of the order the caller iterates metrics in.

    :param metric_names: Metric names that should appear in the chart.
    :return: Mapping ``{metric_name: hex_color}``.
    """
    return {
        name: _PALETTE[i % len(_PALETTE)]
        for i, name in enumerate(sorted(metric_names))
    }


def short_name(metric_name: str) -> str:
    """
    Return a metric's display name with the dimension prefix stripped.

    :param metric_name: Full metric name (e.g. ``completeness_nullRatio``).
    :return: The suffix after the first underscore (e.g. ``nullRatio``).
    """
    parts = metric_name.split("_", 1)
    return parts[1] if len(parts) > 1 else metric_name
