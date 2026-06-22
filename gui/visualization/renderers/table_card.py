"""Big-number card for TABLE-granularity scalar metrics."""
from __future__ import annotations

import streamlit as st

from theme import COLOR_BAD, COLOR_GOOD, COLOR_WARN

_GOOD_COLOR: str = COLOR_GOOD
_WARN_COLOR: str = COLOR_WARN
_BAD_COLOR: str = COLOR_BAD
_NEUTRAL_COLOR: str = "#888888"

_GOOD_THRESHOLD: float = 0.8
_WARN_THRESHOLD: float = 0.5


def render(results: list[dict], dataset_cols: list[str]) -> None:
    """
    Render one big-number card per result with an optional details expander.

    :param results: Result dicts produced by ``get_table_results``.
    :param dataset_cols: Column names of the original dataset (currently unused but
        kept for signature symmetry with other renderers).
    :return: None.
    """
    if not results:
        st.warning("No results to display.")
        return

    for r in results:
        value = r.get("DQvalue")
        explanation = r.get("DQexplanation") or {}

        color = _score_color(value)
        pct = f"{value:.1%}" if value is not None else "—"

        st.markdown(
            f"<div style='text-align:center; padding:6px 10px; background:{color}22; "
            f"border-left:4px solid {color}; border-radius:4px; margin-bottom:4px;'>"
            f"<span style='font-size:1.1rem; font-weight:600; color:{color};'>{pct}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if explanation:
            with st.expander("Details", expanded=False):
                for k, v in explanation.items():
                    st.write(f"**{k}**: {v}")


def _score_color(value: float | None) -> str:
    """Map a 0-1 score to one of three discrete band colors (or grey for None)."""
    if value is None:
        return _NEUTRAL_COLOR
    if value >= _GOOD_THRESHOLD:
        return _GOOD_COLOR
    if value >= _WARN_THRESHOLD:
        return _WARN_COLOR
    return _BAD_COLOR
