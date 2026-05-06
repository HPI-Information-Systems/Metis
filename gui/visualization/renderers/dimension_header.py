"""Compact KPI strip shown at the top of each dimension tab.

Aggregates multiple metric summaries into one row of cards so the user sees the
dimension-level score before any per-metric detail.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from theme import COLOR_BAD, COLOR_GOOD, COLOR_WARN

_BAR_COLOR_GOOD: str = COLOR_GOOD
_BAR_COLOR_WARN: str = COLOR_WARN
_BAR_COLOR_BAD: str = COLOR_BAD
_BAR_BACKGROUND: str = "#e0e0e0"
_BAR_HEIGHT_PX: int = 12

_GOOD_THRESHOLD: float = 0.8
_WARN_THRESHOLD: float = 0.5


def render(
    dimension: str,
    metric_summaries: dict[str, dict],
) -> None:
    """
    Render a dimension header strip.

    :param dimension: Dimension name (e.g. ``"Completeness"``).
    :param metric_summaries: ``{metric_name: summary_dict}`` as returned by
        ``dispatch._cached_metric_summary``. Only active/visible metrics.
    :return: None.
    """
    means = [
        s["mean_score"] for s in metric_summaries.values()
        if s.get("mean_score") is not None
    ]
    total_count = sum(int(s.get("count", 0) or 0) for s in metric_summaries.values())
    n_metrics = len(metric_summaries)
    dim_mean = sum(means) / len(means) if means else None

    c1, c2, c3 = st.columns(3)
    c1.metric(
        f"{dimension} · mean",
        f"{dim_mean:.1%}" if dim_mean is not None else "—",
    )
    c2.metric("Metrics", f"{n_metrics}")
    c3.metric("Results", f"{total_count:,}")

    if dim_mean is not None:
        st.altair_chart(_score_bar(dim_mean), width="stretch")


@st.cache_data(show_spinner=False)
def _score_bar(mean_score: float) -> alt.Chart:
    color = _score_color(mean_score)
    df = pd.DataFrame({"x0": [0.0], "x1": [mean_score], "score": [mean_score]})
    bar = (
        alt.Chart(df)
        .mark_bar(height=_BAR_HEIGHT_PX, cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X("x0:Q", scale=alt.Scale(domain=[0, 1]), axis=None),
            x2=alt.X2("x1:Q"),
            color=alt.value(color),
            tooltip=[alt.Tooltip("score:Q", format=".1%", title="Mean score")],
        )
    )
    bg = (
        alt.Chart(pd.DataFrame({"x0": [0.0], "x1": [1.0]}))
        .mark_bar(
            height=_BAR_HEIGHT_PX,
            color=_BAR_BACKGROUND,
            cornerRadiusTopRight=4,
            cornerRadiusBottomRight=4,
        )
        .encode(x=alt.X("x0:Q", scale=alt.Scale(domain=[0, 1]), axis=None), x2=alt.X2("x1:Q"))
    )
    return (bg + bar).properties(height=20).configure_view(strokeWidth=0)


def _score_color(value: float) -> str:
    """Map a 0-1 score to one of three discrete band colors."""
    if value >= _GOOD_THRESHOLD:
        return _BAR_COLOR_GOOD
    if value >= _WARN_THRESHOLD:
        return _BAR_COLOR_WARN
    return _BAR_COLOR_BAD
