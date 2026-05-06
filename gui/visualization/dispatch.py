"""Route a metric to a single standardized chart, fetching pre-aggregated data only.

One chart per metric, picked from granularity + signals (no metric-name lookups):

- column granularity → per-column bar (direct DQvalues)
- cell granularity   → per-column bar (column-mean aggregates)
- row granularity    → distribution histogram (or pass/fail when binary)
- table granularity  → FD violations bar (when explanation has rule keys), else KPI card

The pipeline still fetches ≤200 pre-aggregated rows per chart from indexed SQL
views, so each metric renders in milliseconds regardless of dataset size.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.result_store import ResultStore
from visualization.renderers import (
    column_bar,
    fd_violations,
    row_distribution,
    table_card,
)
from visualization.signals import extract_signals


@st.cache_data(show_spinner=False)
def _cached_metric_summary(_store, tag: str, metric: str, granularity: str | None = None) -> dict:
    return _store.get_metric_summary(tag, metric, granularity)


@st.cache_data(show_spinner=False)
def _cached_histogram(_store, tag: str, metric: str, granularity: str | None = None) -> pd.DataFrame:
    return pd.DataFrame(_store.get_histogram_data(tag, metric, granularity))


@st.cache_data(show_spinner=False)
def _cached_column_agg(_store, tag: str, metric: str) -> pd.DataFrame:
    return pd.DataFrame(_store.get_column_aggregates(tag, metric))


@st.cache_data(show_spinner=False)
def _cached_column_results(_store, tag: str, metric: str) -> pd.DataFrame:
    return pd.DataFrame(_store.get_column_results(tag, metric))


@st.cache_data(show_spinner=False)
def _cached_worst(_store, tag: str, metric: str, granularity: str | None = None) -> pd.DataFrame:
    return pd.DataFrame(_store.get_worst_results(tag, metric, granularity))


@st.cache_data(show_spinner=False)
def _cached_table_results(_store, tag: str, metric: str) -> list[dict]:
    return _store.get_table_results(tag, metric)


_CACHED_FETCHERS = (
    _cached_metric_summary,
    _cached_histogram,
    _cached_column_agg,
    _cached_column_results,
    _cached_worst,
    _cached_table_results,
)


def invalidate_caches() -> None:
    """Clear all per-(tag, metric) caches. Call after any store write."""
    for fetcher in _CACHED_FETCHERS:
        fetcher.clear()


def render(
    store: ResultStore,
    tag: str,
    metric_name: str,
    dataset_cols: list[str],
    key_prefix: str = "",
) -> None:
    """
    Render the single standardized chart for a metric.

    :param store: The active result store.
    :param tag: The experiment tag.
    :param metric_name: Metric to render.
    :param dataset_cols: Column names of the original dataset.
    :param key_prefix: Stable cache key prefix.
    :return: None.
    """
    if not tag or not metric_name:
        st.warning(f"No results for **{metric_name}**.")
        return

    s = _cached_metric_summary(store, tag, metric_name)
    if s["count"] == 0:
        st.warning(f"No results for **{metric_name}**.")
        return

    primary = s["primary_granularity"]
    cache_key = f"{key_prefix}{tag}::{metric_name}"

    if primary == "column":
        col_results_df = _cached_column_results(store, tag, metric_name)
        column_bar.render(_normalize_column_results(col_results_df), cache_key)
        return

    if primary == "cell":
        col_agg_df = _cached_column_agg(store, tag, metric_name)
        column_bar.render(_normalize_column_agg(col_agg_df), cache_key)
        return

    if primary == "row":
        hist_df = _cached_histogram(store, tag, metric_name)
        signals = extract_signals(s, hist_df)
        row_distribution.render(hist_df, cache_key, is_binary=signals.is_binary)
        return

    if primary == "table":
        table_results = _cached_table_results(store, tag, metric_name)
        hist_df = _cached_histogram(store, tag, metric_name)
        signals = extract_signals(s, hist_df)
        if signals.has_colon_keys:
            rules = fd_violations.extract_rules(table_results)
            fd_violations.render_violations_table(rules, cache_key)
        else:
            table_card.render(table_results, dataset_cols)
        return

    st.info(f"No display available for **{metric_name}**.")


def _normalize_column_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert per-column DQResult rows into ``[column, score, explanation]`` form.

    :param df: DataFrame from ``_cached_column_results``.
    :return: A normalized DataFrame for ``column_bar``.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["column", "score", "explanation"])
    return pd.DataFrame({
        "column": df["column"],
        "score": df["DQvalue"],
        "explanation": df["DQexplanation"].apply(_format_explanation),
    })


def _normalize_column_agg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert cell-level column aggregates into ``[column, score, explanation]`` form.

    :param df: DataFrame from ``_cached_column_agg``.
    :return: A normalized DataFrame for ``column_bar``.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["column", "score", "explanation"])
    return pd.DataFrame({
        "column": df["column"],
        "score": df["mean_score"],
        "explanation": df.apply(
            lambda r: f"{int(r['cnt']):,} cells · σ={float(r['std_score']):.3f}"
            if pd.notna(r.get("std_score")) else f"{int(r['cnt']):,} cells",
            axis=1,
        ),
    })


def _format_explanation(expl) -> str:
    """Format an explanation dict as a single inline string for tooltips."""
    if isinstance(expl, dict) and expl:
        return "  ".join(f"{k}: {v}" for k, v in expl.items() if v is not None)
    return "—"
