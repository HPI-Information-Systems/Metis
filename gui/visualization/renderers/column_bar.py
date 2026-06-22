"""Single horizontal bar chart of mean DQ score per column."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from visualization.color import DQ_COLOR_SCALE


def render(
    df: pd.DataFrame,
    cache_key: str = "",
) -> None:
    """
    Render a horizontal bar chart of ``score`` per ``column``.

    The caller normalizes input into a frame with at least the columns
    ``column`` and ``score`` (0–1 scale). An optional ``explanation`` column is
    used for tooltips when present.

    :param df: Normalized dataframe.
    :param cache_key: Stable cache key for the chart builder.
    :return: None.
    """
    if df is None or df.empty:
        st.warning("No results to display.")
        return

    plot_df = df.copy()
    if "explanation" not in plot_df.columns:
        plot_df["explanation"] = ""
    plot_df = plot_df.sort_values("score", ascending=True)

    st.altair_chart(_chart_cached(cache_key, plot_df), width="stretch")


@st.cache_data(show_spinner=False)
def _chart_cached(cache_key: str, _df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "score:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", title="DQ Score"),
            ),
            y=alt.Y("column:N", sort=None, title="Column"),
            color=alt.Color("score:Q", scale=DQ_COLOR_SCALE, legend=None),
            tooltip=[
                alt.Tooltip("column:N", title="Column"),
                alt.Tooltip("score:Q", format=".1%", title="Score"),
                alt.Tooltip("explanation:N", title="Details"),
            ],
        )
        .properties(height=alt.Step(28))
    )
