"""Temporal comparison: mean DQ score across run timestamps.

Supports two modes:

- single metric, multi-column: one line per column (the legacy view)
- multiple metrics, aggregated: one line per metric (columns collapsed to mean)
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from visualization.metric_palette import metric_colors, short_name

_MAX_COLUMNS_DEFAULT: int = 8
_CHART_HEIGHT_PX: int = 280
_SINGLE_LINE_COLOR: str = "#1f77b4"


def render(results: list[dict], metric_name: str) -> None:
    """
    Render the single-metric temporal view: per-column lines across runs.

    :param results: Pre-aggregated rows from ``load_temporal_data()``, one row
        per ``(experiment_tag, column)`` with keys ``{timestamp, tag, column, DQvalue}``.
    :param metric_name: Metric being charted (used in the title).
    :return: None.
    """
    if not results:
        st.warning(f"No historical data found for **{metric_name}**.")
        return

    df = pd.DataFrame(results)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")

    is_multi_col = df["column"].nunique() > 1 or df["column"].iloc[0] != "(table)"

    if is_multi_col:
        all_cols = sorted(df["column"].unique().tolist())
        selected = st.multiselect(
            "Columns to compare",
            all_cols,
            default=all_cols[:min(_MAX_COLUMNS_DEFAULT, len(all_cols))],
            key=f"temporal_cols_{metric_name}",
        )
        if not selected:
            st.info("Select at least one column.")
            return
        df = df[df["column"].isin(selected)]

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "timestamp:T",
                title="Run timestamp",
                axis=alt.Axis(labelAngle=-30, format="%b %d %H:%M"),
            ),
            y=alt.Y(
                "DQvalue:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
                title="Mean DQ Score",
            ),
            color=(
                alt.Color("column:N", title="Column")
                if is_multi_col
                else alt.value(_SINGLE_LINE_COLOR)
            ),
            tooltip=[
                alt.Tooltip("tag:N", title="Run tag"),
                alt.Tooltip("timestamp:T", title="Timestamp", format="%Y-%m-%d %H:%M"),
                alt.Tooltip("column:N", title="Column"),
                alt.Tooltip("DQvalue:Q", format=".1%", title="Score"),
            ],
        )
        .properties(title=f"Temporal trend: {metric_name}", height=_CHART_HEIGHT_PX)
    )

    st.altair_chart(chart, width='stretch')


def render_multi_metric(
    per_metric_results: dict[str, list[dict]],
    title: str = "Temporal trend across runs",
) -> None:
    """
    Render the multi-metric temporal view: one line per metric, columns collapsed to per-run mean.

    :param per_metric_results: ``{metric_name: temporal_rows}`` where each list is
        the output of ``load_temporal_data()`` for that metric. Columns are
        averaged per run so each metric contributes one line.
    :param title: Chart title.
    :return: None.
    """
    frames: list[pd.DataFrame] = []
    for metric_name, rows in per_metric_results.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        agg = (
            df.groupby(["tag", "timestamp"], as_index=False)["DQvalue"]
            .mean()
        )
        agg["metric"] = short_name(metric_name)
        agg["metric_full"] = metric_name
        frames.append(agg)

    if not frames:
        st.info("No historical data available for the selected metric(s).")
        return

    data = pd.concat(frames, ignore_index=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="ISO8601")

    metric_names = sorted(per_metric_results.keys())
    color_map = metric_colors(metric_names)
    domain = [short_name(m) for m in metric_names]
    color_range = [color_map[m] for m in metric_names]

    chart = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "timestamp:T",
                title="Run timestamp",
                axis=alt.Axis(labelAngle=-30, format="%b %d %H:%M"),
            ),
            y=alt.Y(
                "DQvalue:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
                title="Mean DQ Score",
            ),
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(domain=domain, range=color_range),
                title="Metric",
            ),
            tooltip=[
                alt.Tooltip("metric_full:N", title="Metric"),
                alt.Tooltip("tag:N", title="Run tag"),
                alt.Tooltip("timestamp:T", title="Timestamp", format="%Y-%m-%d %H:%M"),
                alt.Tooltip("DQvalue:Q", format=".1%", title="Score"),
            ],
        )
        .properties(title=title, height=_CHART_HEIGHT_PX)
    )

    st.altair_chart(chart, width="stretch")
