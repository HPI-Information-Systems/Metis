"""Generic grouped bar chart for dimension comparisons.

X-axis = the breakdown sub-unit (column name, FD descriptor, etc.); within each
x-tick, one adjacent bar per metric so the user can compare metrics directly on
the same sub-unit. Y-axis is fixed to 0-100%.

Used by every dimension's comparison view (cell, column, table-with-rules) so
the visual language stays uniform across dimensions.

A metric is only shown if its sub-units overlap with at least one other
metric's. Metrics with entirely unique sub-units (e.g. ``countFDViolations``'s
FD descriptors when the other selected metrics use column names) would sit
alone on the x-axis with no comparison value, so we drop them and surface the
exclusion via a caption.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from visualization.metric_palette import metric_colors, short_name


def render(
    per_metric_rows: dict[str, list[dict]],
    cache_key: str = "",
    sub_order: list[str] | None = None,
) -> None:
    """
    Render the grouped breakdown chart.

    :param per_metric_rows: ``{metric_name: [{"sub": "...", "DQvalue": 0.97}, ...]}``.
        Each metric's list contains one dict per sub-unit (column, FD
        descriptor, …) with the score for that sub-unit.
    :param cache_key: Stable cache key for the chart builder.
    :param sub_order: Optional explicit ordering for the x-axis sub-units. If
        ``None``, sub-units are sorted alphabetically. Sub-units in the data
        but missing from ``sub_order`` are appended at the end.
    :return: None.
    """
    if not per_metric_rows:
        st.caption("No comparable scores.")
        return

    # When more than one metric is supplied, require each shown metric to share
    # at least one sub-unit with another — otherwise its bars sit alone with
    # nothing to compare against.
    if len(per_metric_rows) > 1:
        sub_sets: dict[str, set[str]] = {
            m: {str(r.get("sub", "")) for r in rows}
            for m, rows in per_metric_rows.items()
        }
        chart_metrics = {
            m: rows
            for m, rows in per_metric_rows.items()
            if any(
                sub_sets[m] & sub_sets[other]
                for other in sub_sets
                if other != m
            )
        }
        excluded = [m for m in per_metric_rows if m not in chart_metrics]
        if excluded:
            st.caption(
                "Excluded - no shared breakdown with the other selected metrics: "
                + ", ".join(short_name(m) for m in excluded)
            )
        if not chart_metrics:
            st.caption(
                "Selected metrics have no overlapping breakdowns - nothing to compare."
            )
            return
    else:
        chart_metrics = per_metric_rows

    rows: list[dict] = []
    for metric_name, sub_rows in chart_metrics.items():
        for r in sub_rows:
            v = r.get("DQvalue")
            if v is None or pd.isna(v):
                continue
            rows.append({
                "metric": short_name(metric_name),
                "metric_full": metric_name,
                "sub": str(r.get("sub", "")),
                "DQvalue": float(v),
            })

    if not rows:
        st.caption("No comparable scores.")
        return

    df = pd.DataFrame(rows)
    metric_names = list(chart_metrics.keys())
    short_names = [short_name(m) for m in metric_names]
    color_map = metric_colors(metric_names)
    color_range = [color_map[m] for m in metric_names]

    present_subs = list(df["sub"].unique())
    if sub_order:
        ordered = [s for s in sub_order if s in present_subs]
        ordered += [s for s in sorted(present_subs) if s not in ordered]
    else:
        ordered = sorted(present_subs)

    st.altair_chart(
        _chart_cached(cache_key, df, ordered, short_names, color_range),
        width="stretch",
    )


@st.cache_data(show_spinner=False)
def _chart_cached(
    cache_key: str,
    _df: pd.DataFrame,
    sub_order: list[str],
    metric_order: list[str],
    color_range: list[str],
) -> alt.Chart:
    return (
        alt.Chart(_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "sub:N",
                title=None,
                sort=sub_order,
                axis=alt.Axis(labelAngle=-30, labelLimit=200),
            ),
            xOffset=alt.XOffset("metric:N", sort=metric_order),
            y=alt.Y(
                "DQvalue:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", title="DQ Score"),
            ),
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(domain=metric_order, range=color_range),
                title="Metric",
            ),
            tooltip=[
                alt.Tooltip("metric_full:N", title="Metric"),
                alt.Tooltip("sub:N", title="Breakdown"),
                alt.Tooltip("DQvalue:Q", format=".1%", title="Score"),
            ],
        )
        .properties(height=380)
    )
