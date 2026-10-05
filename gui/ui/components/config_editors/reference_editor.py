"""File-uploader editor for a metric config's reference field."""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st


def render(field_name: str, default, key: str):
    """
    Render an uploader for a reference field and return the loaded frame.

    Falls back to the existing value when the user has not uploaded anything
    on this rerun, so a reference survives navigation between wizard steps.

    :param field_name: The config field being edited, used as the label.
    :param default: The current value of the field.
    :param key: Streamlit widget key.
    :return: The reference DataFrame, or ``None`` when none is set.
    """
    st.caption(
        "This metric compares your data against reference data. "
        "Upload a CSV to use as the reference."
    )
    uploaded = st.file_uploader(field_name, type=["csv"], key=key)

    cache_key = f"_ref_cache__{key}"

    if uploaded is not None:
        stamp = f"{uploaded.name}::{uploaded.size}"
        if st.session_state.get(f"{cache_key}__stamp") != stamp:
            raw = uploaded.read()
            try:
                frame = pd.read_csv(io.BytesIO(raw), encoding="utf-8")
            except UnicodeDecodeError:
                frame = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
            st.session_state[cache_key] = frame
            st.session_state[f"{cache_key}__stamp"] = stamp

    frame = st.session_state.get(cache_key)
    if frame is None and isinstance(default, pd.DataFrame):
        frame = default

    if frame is not None:
        st.success(f"Reference: {len(frame):,} rows x {len(frame.columns)} columns")

    return frame
