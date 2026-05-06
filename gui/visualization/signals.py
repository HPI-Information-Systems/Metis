"""Signal extraction for the visualization algorithm.

Derives a :class:`VisualizationSignals` dataclass from pre-aggregated metric
data. Pure Python — no Streamlit, no I/O, no metric-name lookups.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Bin index 0  → score range [0.00, 0.05) → "Fail"
# Bin index 19 → score range [0.95, 1.00] → "Pass"
_BINARY_BIN_INDICES: frozenset[int] = frozenset({0, 19})


@dataclass
class VisualizationSignals:
    primary: str
    mean_score: float | None
    pct_perfect: float | None
    count: int

    is_binary: bool

    has_colon_keys: bool
    has_vocab_keys: bool

    explanation_keys: frozenset = field(default_factory=frozenset)

    n_columns: int = 0


def extract_signals(
    summary: dict,
    hist_df: pd.DataFrame,
    col_agg_df: pd.DataFrame | None = None,
    col_results_df: pd.DataFrame | None = None,
) -> VisualizationSignals:
    """
    Derive visualization signals from pre-aggregated metric data.

    :param summary: Output of ``get_metric_summary()`` — has ``count``,
        ``granularities``, ``primary_granularity``, ``explanation_keys``,
        ``mean_score``, ``pct_perfect``.
    :param hist_df: 21-bin score histogram with columns ``[bin_idx, count]``.
    :param col_agg_df: Per-column aggregates for cell metrics (may be ``None``).
    :param col_results_df: Per-column results for column metrics (may be ``None``).
    :return: A :class:`VisualizationSignals` with all derived signals set.
    """
    primary = summary.get("primary_granularity", "table")
    mean_score = summary.get("mean_score")
    pct_perfect = summary.get("pct_perfect")
    count = summary.get("count", 0)
    explanation_keys = frozenset(summary.get("explanation_keys") or [])

    is_binary = _detect_binary(hist_df)

    has_colon_keys = any(":" in k for k in explanation_keys)
    has_vocab_keys = {"TotalNotNullValues", "InVocabValues"}.issubset(explanation_keys)

    n_columns = 0
    if col_agg_df is not None and not col_agg_df.empty:
        n_columns = len(col_agg_df)
    elif col_results_df is not None and not col_results_df.empty:
        n_columns = len(col_results_df)

    return VisualizationSignals(
        primary=primary,
        mean_score=mean_score,
        pct_perfect=pct_perfect,
        count=count,
        is_binary=is_binary,
        has_colon_keys=has_colon_keys,
        has_vocab_keys=has_vocab_keys,
        explanation_keys=explanation_keys,
        n_columns=n_columns,
    )


def _detect_binary(hist_df: pd.DataFrame) -> bool:
    """
    Detect whether a metric's score distribution looks binary (pass/fail).

    Returns True iff every non-zero histogram bin is at index 0 or index 19.

    :param hist_df: Histogram dataframe with columns ``[bin_idx, count]``.
    :return: True for a binary distribution, False otherwise.
    """
    if hist_df.empty:
        return False
    nonzero = hist_df[hist_df["count"] > 0]["bin_idx"]
    if nonzero.empty:
        return False
    return set(nonzero).issubset(_BINARY_BIN_INDICES)
