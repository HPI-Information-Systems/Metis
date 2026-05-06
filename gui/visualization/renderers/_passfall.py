"""Shared pass/fail bar chart used by ``row_distribution`` for binary metrics."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from theme import COLOR_BAD, COLOR_GOOD

# Bin index 0  → score range [0.00, 0.05) → "Fail"
# Bin index 19 → score range [0.95, 1.00] → "Pass"
_FAIL_BIN: int = 0
_PASS_BIN: int = 19

_PASS_COLOR: str = COLOR_GOOD
_FAIL_COLOR: str = COLOR_BAD


@st.cache_data(show_spinner=False)
def passfall_chart(cache_key: str, _hist_df: pd.DataFrame, y_title: str, height: int) -> alt.Chart:
    """
    Build a two-bar pass/fail chart for binary metrics.

    :param cache_key: Stable cache key.
    :param _hist_df: Histogram dataframe with columns ``[bin_idx, count]``.
    :param y_title: Y-axis title.
    :param height: Chart height in pixels.
    :return: A configured Altair chart.
    """
    df = _hist_df.copy()
    total = df["count"].sum()

    pass_count = int(df.loc[df["bin_idx"] == _PASS_BIN, "count"].sum())
    fail_count = int(df.loc[df["bin_idx"] == _FAIL_BIN, "count"].sum())

    rows = [
        {"result": "Pass (1.0)", "count": pass_count, "pct": pass_count / total if total else 0},
        {"result": "Fail (0.0)", "count": fail_count, "pct": fail_count / total if total else 0},
    ]
    pf_df = pd.DataFrame(rows)

    return (
        alt.Chart(pf_df)
        .mark_bar()
        .encode(
            x=alt.X("result:N", axis=alt.Axis(labelAngle=0), title=None),
            y=alt.Y("count:Q", title=y_title),
            color=alt.Color(
                "result:N",
                scale=alt.Scale(
                    domain=["Pass (1.0)", "Fail (0.0)"],
                    range=[_PASS_COLOR, _FAIL_COLOR],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("result:N", title="Result"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("pct:Q", format=".1%", title="Share"),
            ],
        )
        .properties(title="Pass / Fail distribution", height=height)
    )
