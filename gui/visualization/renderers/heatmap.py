"""Multi-metric x column pivot heatmap for comparing metrics side-by-side."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from visualization.color import DQ_COLOR_SCALE
from visualization.metric_palette import short_name


def render(heatmap_df: pd.DataFrame, dataset_cols: list[str]) -> None:
    """
    Render a column x metric heatmap from a pre-aggregated DataFrame.

    :param heatmap_df: Columns ``[dq_metric, column, mean_score]`` —
        ``N_metrics x N_cols`` rows.
    :param dataset_cols: Column ordering hint for the Y axis.
    :return: None.
    """
    if heatmap_df.empty:
        st.info("Heatmap requires at least one column-level metric (e.g. completeness, validity).")
        return

    df = heatmap_df.copy()
    df["metric"] = df["dq_metric"].apply(short_name)
    df = df.rename(columns={"mean_score": "DQvalue"})

    n_cols = df["column"].nunique()
    n_metrics = df["metric"].nunique()

    if n_metrics < 2:
        st.info("Select at least two column-level metrics to compare in the heatmap.")
        return

    heatmap_chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("metric:N", title="Metric", axis=alt.Axis(labelAngle=-25)),
            y=alt.Y("column:N", title="Column", sort=dataset_cols or None),
            color=alt.Color("DQvalue:Q", scale=DQ_COLOR_SCALE, title="DQ Score"),
            tooltip=[
                alt.Tooltip("column:N", title="Column"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("DQvalue:Q", format=".1%", title="Score"),
            ],
        )
        .properties(
            title="Data Quality Heatmap",
            height=max(200, 28 * n_cols),
            width=max(200, 80 * n_metrics),
        )
    )

    text = (
        alt.Chart(df)
        .mark_text(fontSize=11)
        .encode(
            x=alt.X("metric:N"),
            y=alt.Y("column:N", sort=dataset_cols or None),
            text=alt.Text("DQvalue:Q", format=".0%"),
            color=alt.condition(
                "datum.DQvalue > 0.5",
                alt.value("black"),
                alt.value("white"),
            ),
        )
    )

    st.altair_chart(heatmap_chart + text, width='stretch')
