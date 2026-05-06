"""Single 21-bin distribution histogram for ROW/CELL granularity scores."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from visualization.renderers._passfall import passfall_chart

_BIN_DENOMINATOR: float = 20.0
_HISTOGRAM_HEIGHT_PX: int = 260
_BAR_COLOR: str = "#1f77b4"


def render(
    hist_df: pd.DataFrame,
    cache_key: str = "",
    is_binary: bool = False,
    y_title: str = "Rows",
    x_title: str = "Row DQ Score",
) -> None:
    """
    Render a histogram of DQ scores, switching to a pass/fail chart when the
    metric is binary.

    :param hist_df: Histogram dataframe with columns ``[bin_idx, count]``.
    :param cache_key: Stable cache key for the chart builder.
    :param is_binary: When True, render a two-bar pass/fail chart instead.
    :param y_title: Y-axis title.
    :param x_title: X-axis title.
    :return: None.
    """
    if hist_df is None or hist_df.empty:
        st.warning("No results to display.")
        return

    if is_binary:
        st.altair_chart(
            passfall_chart(cache_key, hist_df, y_title, _HISTOGRAM_HEIGHT_PX),
            width="stretch",
        )
        return

    st.altair_chart(_chart_cached(cache_key, hist_df, y_title, x_title), width="stretch")


@st.cache_data(show_spinner=False)
def _chart_cached(
    cache_key: str,
    _hist_df: pd.DataFrame,
    y_title: str,
    x_title: str,
) -> alt.Chart:
    df = _hist_df.copy()
    df["bin_left"] = df["bin_idx"] / _BIN_DENOMINATOR
    df["bin_right"] = (df["bin_idx"] + 1) / _BIN_DENOMINATOR

    return (
        alt.Chart(df)
        .mark_bar(color=_BAR_COLOR, opacity=0.85)
        .encode(
            alt.X(
                "bin_left:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
                title=x_title,
            ),
            alt.X2("bin_right:Q"),
            alt.Y("count:Q", title=y_title),
            tooltip=[
                alt.Tooltip("bin_left:Q", format=".1%", title="Score ≥"),
                alt.Tooltip("bin_right:Q", format=".1%", title="Score <"),
                alt.Tooltip("count:Q", title=y_title),
            ],
        )
        .properties(height=_HISTOGRAM_HEIGHT_PX)
    )
