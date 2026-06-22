"""Config editor for ``timeliness_heinrich`` — per-column nested configuration."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from metis.metric.timeliness.timeliness_heinrich_config import (
    timeliness_heinrich_column_config,
)

_DT_PRECISION_OPTIONS: list[str | None] = [
    None, "year", "month", "day", "hour", "minute", "second", "microsecond",
]

_DATE_KEYWORDS: tuple[str, ...] = ("date", "time", "timestamp", "created", "updated")


def render(config_class, key_prefix: str, df_columns: list[str], df: pd.DataFrame | None = None):
    """
    Render the per-column timeliness configuration editor.

    :param config_class: The ``timeliness_heinrich_config`` dataclass.
    :param key_prefix: Streamlit widget key prefix.
    :param df_columns: Column names of the active dataframe.
    :param df: The active dataframe (used to auto-detect timestamp precision).
    :return: A populated ``config_class`` instance, or ``None`` if incomplete.
    """
    st.caption(
        "Configure timeliness for each column you want to assess. "
        "Only configured columns will produce results."
    )

    cols_key = f"{key_prefix}__columns"
    selected_cols = st.multiselect(
        "Columns to configure",
        options=df_columns,
        key=cols_key,
        help="Select the data columns whose timeliness you want to measure.",
    )

    if not selected_cols:
        st.caption("Select at least one column above.")
        return None

    per_column: dict = {}
    all_valid = True

    for col in selected_cols:
        with st.expander(f"⚙️ Column: **{col}**", expanded=True):
            column_cfg, column_valid = _render_column_block(
                col, df_columns, df, key_prefix,
            )
            if column_cfg is None:
                all_valid = all_valid and column_valid
                continue
            per_column[col] = column_cfg

    if not all_valid or not per_column:
        return None

    try:
        return config_class(timeliness_config_per_column=per_column)
    except (TypeError, ValueError) as exc:
        st.error(f"Config error: {exc}")
        return None


def _render_column_block(
    col: str,
    df_columns: list[str],
    df: pd.DataFrame | None,
    key_prefix: str,
) -> tuple[object | None, bool]:
    """
    Render the form widgets for one configured column and return its column config.

    :param col: Name of the column being configured.
    :param df_columns: Column names of the active dataframe.
    :param df: The active dataframe (used for precision auto-detection).
    :param key_prefix: Streamlit widget key prefix.
    :return: ``(column_config, is_valid)``. ``column_config`` is ``None`` when the
        user-supplied JSON kwargs failed to parse; in that case ``is_valid`` is False.
    """
    decline_rate = st.number_input(
        "Decline rate (λ)",
        min_value=0.0,
        max_value=20.0,
        value=0.5,
        step=0.1,
        key=f"{key_prefix}__{col}__rate",
        help=(
            "Score = e^(−λ × age_years). "
            "λ = 0.5 → ~60% score after 1 year; "
            "λ = 1.0 → ~37% score after 1 year."
        ),
    )

    date_cols = [
        c for c in df_columns
        if any(kw in c.lower() for kw in _DATE_KEYWORDS)
    ]
    default_date_idx = df_columns.index(date_cols[0]) if date_cols else 0
    ingestion_col = st.selectbox(
        "Ingestion date column",
        options=df_columns,
        index=default_date_idx,
        key=f"{key_prefix}__{col}__date_col",
        help="Column that holds the date/timestamp when this value was recorded.",
    )

    sim_date = st.text_input(
        "Simulated assessment date (optional)",
        value="",
        placeholder="e.g. 2024-01-01",
        key=f"{key_prefix}__{col}__sim_date",
        help="If empty, the current date is used.",
    )

    precision = st.selectbox(
        "Timestamp precision override (optional)",
        options=_DT_PRECISION_OPTIONS,
        format_func=lambda x: x or "(auto-detect)",
        key=f"{key_prefix}__{col}__precision",
        help="Override the detected precision of dates in the ingestion date column.",
    )

    dt_kwargs_raw = st.text_input(
        "pandas.to_datetime kwargs (JSON, optional)",
        value="",
        placeholder='e.g. {"format": "%Y-%m-%d"}',
        key=f"{key_prefix}__{col}__dt_kwargs",
    )

    dt_kwargs = None
    if dt_kwargs_raw.strip():
        try:
            dt_kwargs = json.loads(dt_kwargs_raw)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON for `{col}`: {exc}")
            return None, False

    effective_precision = precision
    if effective_precision is None and df is not None and ingestion_col in df.columns:
        column_values = df[ingestion_col]
        if column_values.isna().any() or not column_values.dropna().apply(lambda v: isinstance(v, str)).all():
            effective_precision = "day"
            st.caption(
                f":material/info: Column `{ingestion_col}` contains non-string or missing values — "
                "precision auto-set to **day** to avoid parse errors. "
                "Override above if needed."
            )

    column_cfg = timeliness_heinrich_column_config(
        decline_rate=decline_rate,
        ingestion_date_column=ingestion_col,
        to_datetime_kwargs=dt_kwargs,
        simulated_assessment_date=sim_date.strip() or None,
        simulated_timestamp_precision=effective_precision,
    )
    return column_cfg, True
