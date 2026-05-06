"""Unified worst-N table pivoting multiple metrics' scores onto one row per item.

One table instead of N separate worst-N tables: each row is a cell or row from
the dataset; each metric contributes a score column. Sorted by the minimum
score across active metrics so the real offenders surface first.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from visualization.metric_palette import short_name

_SLIDER_MIN: int = 5
_SLIDER_MAX_CAP: int = 200
_SLIDER_STEP: int = 5
_SCORE_COLUMN_PREFIX: str = "score::"


def render(
    per_metric_worst: dict[str, pd.DataFrame],
    granularity: str,
    cache_key: str = "",
    n_default: int = 20,
) -> None:
    """
    Render a unified worst-N table.

    :param per_metric_worst: ``{metric_name: worst_df}`` where each df has
        columns ``[column, row_index, dq_value]``. The column value may be
        ``"(unknown)"`` for row-only metrics.
    :param granularity: ``"cell"`` or ``"row"`` — drives label and key columns.
    :param cache_key: Stable cache key for the slider state.
    :param n_default: Default slider value.
    :return: None.
    """
    if not per_metric_worst:
        return

    pivot = _pivot(per_metric_worst, granularity)
    if pivot.empty:
        return

    max_n = min(_SLIDER_MAX_CAP, len(pivot))
    n = st.slider(
        f"Worst N {granularity}s",
        min_value=_SLIDER_MIN,
        max_value=max(_SLIDER_MIN, max_n),
        value=min(n_default, max(_SLIDER_MIN, max_n)),
        step=_SLIDER_STEP,
        key=f"multi_worst_{cache_key}",
    )

    shown = pivot.head(n).copy()
    for col in shown.columns:
        if col.startswith(_SCORE_COLUMN_PREFIX):
            label = col.split("::", 1)[1]
            shown[label] = shown[col].map(
                lambda v: "—" if pd.isna(v) else f"{v:.1%}"
            )
            shown.drop(columns=[col], inplace=True)

    label_cols = ["Column", "Row"] if granularity == "cell" else ["Row"]
    metric_cols = [c for c in shown.columns if c not in ("Column", "Row", "_min")]
    ordered = label_cols + metric_cols
    ordered = [c for c in ordered if c in shown.columns]

    st.write(f"**{len(shown)} worst-scoring {granularity}(s):**")
    st.dataframe(
        shown[ordered].reset_index(drop=True),
        width="stretch",
        hide_index=True,
    )


def _pivot(
    per_metric_worst: dict[str, pd.DataFrame],
    granularity: str,
) -> pd.DataFrame:
    """
    Pivot worst-rows from long to wide format, one column per metric.

    :param per_metric_worst: Per-metric worst-row dataframes.
    :param granularity: ``"cell"`` or ``"row"``.
    :return: A wide DataFrame ordered by minimum metric score across columns.
    """
    frames: list[pd.DataFrame] = []
    for metric_name, df in per_metric_worst.items():
        if df is None or df.empty:
            continue
        part = df[["column", "row_index", "dq_value"]].copy()
        part["metric"] = metric_name
        frames.append(part)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)

    if granularity == "cell":
        data["_key"] = list(zip(data["column"], data["row_index"]))
        group_cols = ["column", "row_index"]
    else:
        data["_key"] = data["row_index"]
        group_cols = ["row_index"]

    pivoted = data.pivot_table(
        index=group_cols,
        columns="metric",
        values="dq_value",
        aggfunc="min",
    ).reset_index()

    # Rename metric columns to score::<short_name> markers so the renderer
    # knows which columns to format as percentages.
    score_cols = [c for c in pivoted.columns if c not in group_cols]
    pivoted = pivoted.rename(
        columns={c: f"{_SCORE_COLUMN_PREFIX}{short_name(c)}" for c in score_cols}
    )

    score_only = [c for c in pivoted.columns if c.startswith(_SCORE_COLUMN_PREFIX)]
    pivoted["_min"] = pivoted[score_only].min(axis=1)
    pivoted = pivoted.sort_values("_min", ascending=True)

    pivoted = pivoted.rename(columns={"column": "Column", "row_index": "Row"})

    return pivoted
