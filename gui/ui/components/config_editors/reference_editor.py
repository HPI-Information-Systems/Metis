"""File-uploader editor for a metric config's reference field."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.csv_upload import read_csv_upload
from metis.utils.dataset_checks import DatasetError
from metis.utils.reference_loader import get_default_reference


def render(field_name: str, default, key: str):
    """
    Render an uploader for a reference field and return the loaded frame.

    Falls back to the existing value when the user has not uploaded anything
    on this rerun, so a reference survives navigation between wizard steps.

    When a global reference dataset is active, an upload here overrides it
    for this metric only. Without an upload the field stays ``None`` and the
    metric resolves the global reference itself.

    :param field_name: The config field being edited, used as the label.
    :param default: The current value of the field.
    :param key: Streamlit widget key.
    :return: The reference DataFrame, or ``None`` when none is set.
    """
    global_reference = get_default_reference()
    if global_reference is not None:
        st.caption(
            "This metric compares your data against reference data. It uses "
            "the reference dataset from the Dataset step. Upload a CSV here "
            "to use a different reference for this metric only."
        )
    else:
        st.caption(
            "This metric compares your data against reference data. "
            "Upload a CSV to use as the reference."
        )

    cache_key = f"_ref_cache__{key}"
    version_key = f"{cache_key}__version"
    error_key = f"{cache_key}__error"

    # A file uploader cannot be emptied from code. Bumping the version gives
    # it a fresh key instead, which is how a metric's own reference is
    # dropped again.
    version = st.session_state.get(version_key, 0)
    uploader_key = key if version == 0 else f"{key}__v{version}"
    uploaded = st.file_uploader(field_name, type=["csv"], key=uploader_key)

    if uploaded is not None:
        stamp = f"{uploaded.name}::{uploaded.size}"
        if st.session_state.get(f"{cache_key}__stamp") != stamp:
            st.session_state[f"{cache_key}__stamp"] = stamp
            try:
                upload = read_csv_upload(uploaded.read(), uploaded.name)
            except DatasetError as e:
                # A declined file must not leave an earlier reference in
                # place, or the metric would silently be scored against it.
                st.session_state.pop(cache_key, None)
                st.session_state[error_key] = str(e)
            else:
                st.session_state[cache_key] = upload.frame
                st.session_state.pop(error_key, None)
    elif st.session_state.pop(error_key, None) is not None:
        st.session_state.pop(f"{cache_key}__stamp", None)

    if st.session_state.get(error_key):
        st.error(st.session_state[error_key])

    frame = st.session_state.get(cache_key)
    if frame is None and isinstance(default, pd.DataFrame):
        frame = default

    if frame is not None:
        if global_reference is not None:
            st.success(
                f"Reference for this metric: {len(frame):,} rows x "
                f"{len(frame.columns)} columns. It replaces the reference dataset "
                "from the Dataset step."
            )
            st.button(
                "Use the reference dataset from the Dataset step instead",
                key=f"{key}__use_global",
                on_click=_drop_override,
                args=(cache_key, version_key),
            )
        else:
            st.success(f"Reference: {len(frame):,} rows x {len(frame.columns)} columns")
    elif global_reference is not None:
        st.info(
            f"Using the reference dataset from the Dataset step: "
            f"{len(global_reference):,} rows x "
            f"{len(global_reference.columns)} columns"
        )

    return frame


def _drop_override(cache_key: str, version_key: str) -> None:
    """
    Forget a metric's own reference so it falls back to the global one.

    :param cache_key: Session key holding the metric's uploaded reference.
    :param version_key: Session key holding the uploader's key version.
    :return: None.
    """
    st.session_state.pop(cache_key, None)
    st.session_state.pop(f"{cache_key}__stamp", None)
    st.session_state[version_key] = st.session_state.get(version_key, 0) + 1
